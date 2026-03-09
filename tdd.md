# Technical Design Document (TDD)

## Reinforcement Learning Autonomous Car Trainer

------------------------------------------------------------------------

## System Architecture

Simulation Environment → State Observation → Neural Network Policy →
Action → Physics Engine → Reward → Training Update

The neural network directly outputs actions controlling the vehicle.

------------------------------------------------------------------------

## Simulation Environment

Components:

Car physics model

Track geometry

Collision detection

Sensor system

Physics timestep loop

Suggested libraries:

numpy\
pygame\
torch

------------------------------------------------------------------------

## Car Physics Model

State variables:

position (x, y)

velocity

heading angle

angular velocity

acceleration

A simplified **bicycle model** will be used to simulate steering
dynamics.

------------------------------------------------------------------------

## Sensor System

Ray‑cast sensors detect distance to track boundaries.

Typical layout:

Front sensors\
Side sensors\
Angled sensors

Typical sensor count:

5‑9 sensors

------------------------------------------------------------------------

## Observation State Vector

Example:

\[ distance_sensor_1, distance_sensor_2, distance_sensor_3,
distance_sensor_4, distance_sensor_5, velocity, heading_angle,
angular_velocity, distance_from_center\]

Typical size: 8‑15 inputs.

------------------------------------------------------------------------

## Action Space

Continuous control.

steering ∈ \[-1,1\]

throttle ∈ \[0,1\]

Actions are produced directly by the neural network.

------------------------------------------------------------------------

## Custom Neural Network

The neural network must be **implemented manually in PyTorch**.

Example architecture:

Input Layer

Dense 256\
ReLU

Dense 256\
ReLU

Dense 128\
ReLU

Output Layer

Outputs:

steering

throttle

------------------------------------------------------------------------

## Reinforcement Learning Algorithm

Recommended algorithms:

PPO

SAC

DDPG

Preferred:

PPO (Proximal Policy Optimization)

------------------------------------------------------------------------

## Reward Function Example

Forward motion reward:

+1 \* forward_velocity

Progress along track:

+0.5 \* progress

Crash penalty:

-10

Leaving track:

-5

Time penalty:

-0.01 per timestep

------------------------------------------------------------------------

## Episode Termination

Episode ends if:

Car crashes

Car leaves track

Maximum timestep reached

------------------------------------------------------------------------

## Logging

Metrics logged:

episode reward

episode length

crash rate

training loss

TensorBoard used for visualization.

------------------------------------------------------------------------

## Environment API

Environment follows Gym style interface.

env.reset()

env.step(action)

env.render()

Return values:

state, reward, done, info
