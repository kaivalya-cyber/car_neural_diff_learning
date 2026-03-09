# tasks.md

# RL Autonomous Car Trainer --- Implementation Task List

This file defines the **step‑by‑step execution plan** for building the
project described in:

-   prd.md
-   tdd.md
-   ai_context.md

AI coding agents should execute tasks **in order** and mark them
complete as they finish.

Rules: - Do NOT skip tasks - Do NOT create placeholder implementations -
Each task must produce **fully working code** - After completing a task,
verify code compiles before moving on

------------------------------------------------------------------------

# Phase 1 --- Repository Setup

-   [x] Read and analyze the following files:
    -   prd.md
    -   tdd.md
    -   ai_context.md
-   [x] Verify repository structure matches specification.

Expected folders:

    env/
    agent/
    training/
    visualization/
    configs/
    tests/

-   [x] Create missing folders if they do not exist.

------------------------------------------------------------------------

# Phase 2 --- Environment Core

Goal: Build the simulation before any RL code.

Files to implement:

    env/car.py
    env/physics.py
    env/track.py
    env/sensors.py
    env/environment.py

Tasks:

-   [x] Implement **Car class**
    -   position
    -   velocity
    -   heading
    -   steering angle
    -   throttle
-   [x] Implement **Physics update**
    -   timestep update
    -   bicycle model steering
    -   velocity update
-   [x] Implement **Track system**
    -   boundary representation
    -   wall segments
    -   collision detection
-   [x] Implement **Sensor system**
    -   raycast distance sensors
    -   configurable number of sensors
    -   normalized outputs (0‑1 range)
-   [x] Implement **Environment class**
    -   reset()
    -   step(action)
    -   render()

The environment must return:

    state, reward, done, info

------------------------------------------------------------------------

# Phase 3 --- Observation System

Goal: Build the observation state returned to the agent.

Tasks:

-   [x] Combine sensor readings into state vector

Example:

    state = [
    sensor_1_distance
    sensor_2_distance
    sensor_3_distance
    sensor_4_distance
    sensor_5_distance
    velocity
    heading_angle
    angular_velocity
    distance_from_center
    ]

-   [x] Normalize inputs for neural network use

-   [x] Ensure consistent state vector size

------------------------------------------------------------------------

# Phase 4 --- Neural Network

Files:

    agent/neural_network.py
    agent/policy.py

Tasks:

-   [x] Implement **custom PyTorch neural network**

Architecture example:

    Input Layer

    Dense 256
    ReLU

    Dense 256
    ReLU

    Dense 128
    ReLU

    Output Layer

Outputs:

    steering
    throttle

-   [x] Implement **Policy wrapper**
    -   converts network outputs to environment actions
    -   clamps action values within limits

------------------------------------------------------------------------

# Phase 5 --- Reward System

File:

    training/reward_function.py

Tasks:

-   [x] Implement reward calculation

Example reward terms:

Forward motion reward

    +1 * forward_velocity

Track progress reward

    +0.5 * progress_along_track

Crash penalty

    -10

Leaving track penalty

    -5

Time penalty

    -0.01 per step

-   [x] Ensure reward returns scalar float

------------------------------------------------------------------------

# Phase 6 --- Reinforcement Learning Training

Files:

    training/trainer.py
    training/training_loop.py

Tasks:

-   [x] Implement **training loop**

Example structure:

    for episode in range(num_episodes):

        state = env.reset()

        while not done:

            action = policy(state)

            next_state, reward, done, info = env.step(action)

            store_transition()

            update_policy()

            state = next_state

-   [x] Implement **PPO algorithm**

Components:

-   policy network

-   value network

-   advantage estimation

-   gradient updates

-   [x] Add **experience buffer**

-   [x] Add **policy update step**

------------------------------------------------------------------------

# Phase 7 --- Visualization

Files:

    visualization/renderer.py

Tasks:

-   [x] Render track boundaries

-   [x] Render car position and orientation

-   [x] Render sensor rays

-   [x] Display debugging info

    -   episode reward
    -   timestep count
    -   speed

------------------------------------------------------------------------

# Phase 8 --- Logging

Tasks:

-   [x] Log training metrics

Metrics:

    episode_reward
    episode_length
    crash_rate
    training_loss

-   [x] Integrate TensorBoard logging

------------------------------------------------------------------------

# Phase 9 --- Configuration System

File:

    configs/hyperparameters.yaml

Tasks:

-   [x] Define hyperparameters

Example:

    learning_rate: 0.0003
    gamma: 0.99
    batch_size: 64
    num_episodes: 10000

-   [x] Ensure training loop reads configuration file

------------------------------------------------------------------------

# Phase 10 --- Integration

Tasks:

-   [x] Connect environment + neural network + trainer

-   [x] Implement main execution script

File:

    main.py

Main script must support:

    train mode
    evaluate mode
    render simulation

------------------------------------------------------------------------

# Phase 11 --- Testing

Files:

    tests/test_environment.py
    tests/test_reward.py
    tests/test_network.py

Tasks:

-   [x] Validate environment reset()

-   [x] Validate sensor outputs

-   [x] Validate neural network forward pass

------------------------------------------------------------------------

# Completion Criteria

The project is considered complete when:

-   Simulation runs successfully
-   Training loop executes without errors
-   Agent improves reward over time
-   Trained model can drive without crashing

------------------------------------------------------------------------

# Phase 12 --- Vectorized Environments

-   [x] Implement `VectorEnv` using multiprocessing
-   [x] Update `RacingPolicy` to handle batches
-   [x] Update `PPOTrainer.update` logic and `Memory.rewards`
-   [x] Update `main.py` training loop for vectorized handling
-   [x] Write unit tests for vectorized step mapping

------------------------------------------------------------------------

# Phase 13 --- Procedural Track Generation

-   [x] Modify `Track` class to generate spline-based tracks
-   [x] Ensure `outer_boundary` and `inner_boundary` are collision-safe
-   [x] Create random starting position picking algorithm
-   [x] Modify `CarEnv` to regenerate track on each reset
-   [x] Write unit tests for procedural generation

------------------------------------------------------------------------

# Phase 14 --- Curriculum Learning

-   [x] Create `CurriculumManager` to track performance and map levels to generation params
-   [x] Update `Track` and `CarEnv` to accept difficulty parameters
-   [x] Modify `VectorEnv` to broadcast difficulty parameters to worker processes
-   [x] Integrate `CurriculumManager` into `training_loop.py` and log diff level
-   [x] Add curriculum unit test

------------------------------------------------------------------------

# Phase 15 --- GPU Acceleration

-   [x] Implement dynamic device detection (`cuda`, `mps`, `cpu`) in `training_loop.py` and `main.py`
-   [x] Modify tensor creation in `trainer.py` to target the `device` directly, avoiding CPU memory copies
-   [ ] Ensure `VectorEnv` tensors transfer efficiently to the hardware
-   [x] Verify the training and evaluation loops run without device mismatch errors

------------------------------------------------------------------------

# Phase 16 --- Live Dashboards and Visualization

-   [x] Modify `PPOTrainer.update` to return average `policy_loss` and `value_loss`
-   [x] Update `training_loop.py` to log losses to TensorBoard
-   [x] Update `training_loop.py` to continuously append to `logs/metrics.csv`
-   [x] Create `visualization/dashboard.py` with asynchronous live `matplotlib` charts

------------------------------------------------------------------------

# Phase 17 --- Automated Hyperparameter Tuning

-   [x] Define search grid in `configs/tune.yaml`
-   [x] Implement Cartesian product grid search pipeline in `training/tuner.py`
-   [x] Log evaluation metrics across multiple fresh agent instances
-   [x] Export optimal sets intelligently into `configs/tuning_results.yaml`
-   [x] Integrate `--mode tune` into `main.py`

------------------------------------------------------------------------

# Phase 18 --- Multi-GPU Acceleration

-   [x] Add `torch.cuda.device_count()` checks
-   [x] Apply `torch.nn.DataParallel` wrapper onto Policy and Value networks
-   [x] Verify the `get_action` and `update` loops still consume nested batches correctly
