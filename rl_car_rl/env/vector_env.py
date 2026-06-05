import multiprocessing as mp
import numpy as np
import sys
import os

from env.environment import CarEnv

def worker(remote, parent_remote, sensor_count=16, obstacle_count=0, track_type="procedural"):
    parent_remote.close()
    env = CarEnv(sensor_count=sensor_count, obstacle_count=obstacle_count, track_type=track_type)
    
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            if done:
                # Store terminal observation in info for value estimation
                info['terminal_observation'] = obs
                obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            obs = env.reset()
            remote.send(obs)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            # Not strictly using gym spaces object, so we return dim
            remote.send((9, 2))
        elif cmd == 'set_difficulty':
            env.set_difficulty(data)
            remote.send(True)
        else:
            raise NotImplementedError

class VectorEnv:
    """
    Vectorized Environment wrapper that runs multiple environments 
    in parallel using Python's multiprocessing.
    """
    def __init__(self, num_envs=64, sensor_count=16, obstacle_count=0, track_type="procedural"):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            p = mp.Process(target=worker, args=(work_remote, remote, sensor_count, obstacle_count, track_type))
            p.daemon = True
            p.start()
            self.processes.append(p)
            
        for remote in self.work_remotes:
            remote.close()
            
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
            
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def step(self, actions):
        """
        actions: list or array of shape (num_envs, action_dim)
        """
        actions = np.array(actions)
        if actions.ndim == 1:
            actions = np.expand_dims(actions, 0)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def set_difficulty(self, params):
        for remote in self.remotes:
            remote.send(('set_difficulty', params))
            
        # Wait for acks
        for remote in self.remotes:
            remote.recv()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()
