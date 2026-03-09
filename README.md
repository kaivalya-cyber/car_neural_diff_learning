# Autonomous Racing RL (Neural Tesla)

## 📌 Project Overview
**Autonomous Racing RL (Neural Tesla)** is an advanced Reinforcement Learning environment built entirely from scratch in Python. It is designed to train autonomous vehicles to navigate mathematically generated tracks using the Proximal Policy Optimization (PPO) algorithm.

The purpose of this project is to simulate vehicular autonomous control, requiring the agent to utilize simulated raycast sensors, a continuous action space (steering and throttle), and dynamic physics models to successfully traverse procedural racing circuits without crashing or straying out of bounds.

---

## 🌟 Key Features
- **Custom 2D Physics Engine:** Accurate kinematics mapping bicycle heading sweeps, variable velocities, and precise boundary collision arrays.
- **PPO Neural Network:** Fully dense custom multi-layer perceptron generating Normal distributions over steering/throttle with backpropagated Value Network estimations.
- **Vectorized Environments:** Multiprocessing workers executing isolated RL environments synchronously for batched simulation updates.
- **Procedural Track Generation:** Infinitely unique tracks generated per episode using Catmull-Rom spline pathing algorithms with adjustable variances and bounding box geometry.
- **Curriculum Learning:** Dynamic threshold escalation based on agent rolling-return performance scaling tracks from wide/gentle (Level 0.0) to tight/jagged (Level 1.0).
- **GPU & Multi-GPU Execution:** Dynamic `cuda`/`mps` PyTorch memory allocation coupled with `torch.nn.DataParallel` distributing neural inferences and backward passes flawlessly across multiple NVIDIA pipelines dynamically.
- **Automated Hyperparameter Tuning:** Complete CLI wrapper iterating Cartesian grid combinations spanning multi-generational configurations.
- **Live Dashboards:** Asynchronous external data hooks plotting dynamic Matplotlib evaluation graphs while TensorBoard digests PPO gradients.

---

## ⚙️ System Architecture
The architecture cleanly separates the execution environment, the RL agent, the training hardware loops, and the external data analytics pipelines.

### Environment & Simulation (`env/`)
A custom 2D Pygame mathematical engine handles collision arrays, vehicle properties, physics timestep integrations, and sensor polling. Action step transitions are handled rapidly without rendering penalties, allowing the simulation to achieve massive parallel speed multipliers.

### Agent & Networks (`agent/`)
Uses standard Multi-Layer Perceptrons predicting parameters for continuous action spaces. The PPO Algorithm uses the Actor-Critic methodology, instantiating a `RacingPolicy` network predicting normal distributions for the environment actuators, and a Value Network mapping the reward estimations given the 9-dimensional sensor array batch outputs.

### Training & GPU Pipeline (`training/`)
`PPOTrainer` computes generalized advantage estimations over trajectory rollouts. `VectorEnv` intercepts environment iterations passing batched tensors sequentially back to PyTorch `DataParallel` clusters resulting in hardware-integrated tensor manipulation strictly kept off the system CPU.

---

## 🏗 Core Components

- **Environment System (`environment.py`)**: Wraps step, reset, and render loops into an OpenAI Gym-like API interface.
- **Physics Model (`physics.py`)**: Implements Euler integrations governing 2D acceleration, simulated drag, and steering heading shifts natively mapped off the bicycle kinematic model.
- **Sensor Model (`sensors.py`)**: Casts normalized linear segments outward tracking intersections natively against the spline boundaries giving the agent 5 directional perception bands ranging from `0.0` to `1.0`.
- **Procedural Track System (`track.py`)**: Calculates mathematically smooth inner/outer boundaries mapping random control points with continuous radius scaling.
- **Neural Network (`neural_network.py`)**: Predicts continuous policy deviations mapping states to throttle and steering vectors natively on CUDA/MPS descriptors.
- **Reward System (`training/reward_function.py` - implicit)**: Computes immediate localized feedback. High rewards for velocity bounds and track progress while heavily penalizing boundary collision.
- **Visualization System (`renderer.py`)**: Decoupled Pygame rendering pipeline visually constructing the agent, sensory raycasts, bounding paths, and on-screen metrics selectively.
- **Logging System (`training_loop.py`)**: Handles periodic Tensorboard ingestion, checkpoint tracking, and decoupling async outputs to external dashboards.

---

## 🚀 Advanced Upgrades

### Vectorized Environments
Operating Python sequential environments restricts the PyTorch architecture to learning single batches concurrently. By utilizing `multiprocessing.Process` via `env/vector_env.py`, the main loop instantiates 32-128 completely isolated asynchronous sub-processes that parse commands via `Connection.recv()`. When the master calls `step()`, all environments update parallelly giving PyTorch dense batched tracking inputs spanning memory arrays.

### Procedural Track Generation
To prevent the agent from generalizing over single monolithic maps, the track geometry uses Catmull-Rom splines randomly shifting control points around the bounds of the track dimension space. Valid outer nodes and internal margins construct continuous loop paths randomized identically on each reset. The agent must generalize actual driving protocols natively.

### Curriculum Learning
Rather than paralyzing the agent against mathematically generated narrow spline tracks natively, the `CurriculumManager` dynamically modifies the trajectory generation parameters. 
Level 0.0 constructs 120px wide tracks relying on 8 basic control points producing gentle sweeping bounds. 
Level 1.0 constructs 60px tight boundaries over 16 varied control points for jagged hairpin tracks globally updated via the rolling mean agent score!

### GPU PPO Training
Python natively computes arrays in numpy lists consuming massive main-board RAM overhead and restricting processing speeds to sequential core-clocks. The `PPOTrainer` overrides numpy buffers dynamically converting trajectory rewards, state spaces, advantages, and tensors structurally over active `torch.device()` pools allocating memory cleanly on VRAM.

### Multi-GPU Training
Single-node gradient synchronization suffers immensely processing thick vectorized environments via heavy iteration passes. `torch.nn.DataParallel` natively wraps the PyTorch core networks dynamically resolving node-count limits in `cuda`. If multiple GPUs exist, observations are chunked across the GPU grid equally before resolving to the shared master node `cuda:0` reducing PPO update delays heavily.

### Hyperparameter Tuning
Optimizing hyperparameter combinations conventionally is tedious. `training/tuner.py` executes a mathematical Cartesian product over vectors inside `configs/tune.yaml` running independent testing permutations sequentially dynamically grading final curriculum scaling and returning outputs neatly compiled over `configs/tuning_results.yaml`.

### Training Dashboards
To eliminate threading pauses locking up the primary model trainer during rendering sequences, visualization runs entirely decoupled externally! `visualization/dashboard.py` runs Python's matplotlib async looping `logs/metrics.csv` reading generated scalars visually graphing the 100-episode Reward Moving Average vs the target Curriculum Difficulty level. 

---

## 🔄 Training Pipeline

1. Configuration YAML loaded. Multiprocessing `VectorEnv` spawner initialized creating the 32 discrete background environments.
2. `state = env.reset()` commands distributed asynchronously generating 32 procedural maps synchronously over starting Level thresholds natively.
3. Batches parsed securely to PyTorch generating normal distributions (Policy Outputs) executed against the models inside Simulation instances.
4. Next actions sequentially generated recursively appending batched trajectory transitions to PyTorch decoupled CPU memory lists.
5. If an environment crashes, the reward logs natively internally and randomly regenerated maps override the target memory arrays resetting parallel arrays explicitly.
6. Curriculum evaluated periodically adjusting environment generation rules internally globally passing difficulty factors.
7. Trajectory arrays flushed securely to GPU Memory triggering PPO Update routines mapping `loss.backward()` over dense TensorBoard statistics cleanly!

---

## ⚡ Performance Optimizations

- **Vectorized Simulation**: Isolated agent instances resolving Physics boundaries massively decreasing single-threading Pygame iteration delays.
- **GPU Acceleration**: VRAM native architecture decoupling arrays from NumPy RAM minimizing the I/O bottleneck over the motherboard PCIe constraints.
- **Multi-GPU Parallelization**: Tensor distribution maps evenly tracking gradients perfectly symmetrically across node topologies automatically tracking to best configurations natively mapping models structurally.
- **Curriculum Ramps**: Bounding early neural networks over wide maps generating positive feedback recursively preventing total chaotic random weight drops saving massive hours of CPU cycles manually mapping curves visually.

---

## 📂 Repository Structure

```
rl_car_rl/
├── agent/
│   ├── neural_network.py      # Core PyTorch MLP layers
│   └── policy.py              # RacingPolicy generating mapped Normal Distribution choices
├── configs/
│   ├── hyperparameters.yaml   # Primary PPO training configurations
│   ├── tune.yaml              # Hyperparameter Tuner configuration grid
│   └── tuning_results.yaml    # Generated output from Grid Search 
├── env/
│   ├── car.py                 # Primary Vehicle physics class
│   ├── environment.py         # OpenAI Gym wrapping target implementation
│   ├── physics.py             # Bicycle-model Euler calculation layers
│   ├── sensors.py             # Spline bounding line-segment calculators
│   ├── track.py               # Generative Procedural Geometry builder 
│   └── vector_env.py          # Multiprocessing node wrapper 
├── logs/                      # Dynamic output tensorboards
├── tests/                     # Automated unittest suite validating integration stability 
├── training/
│   ├── curriculum.py          # Dynamic generative configuration mapping threshold manager 
│   ├── trainer.py             # Central PPO loop utilizing dense batched matrices internally
│   ├── training_loop.py       # Main system hook executing recursive arrays visually 
│   └── tuner.py               # Decoupled gridsearch instantiation node
├── visualization/
│   ├── dashboard.py           # Decoupled matplotlib async telemetry
│   └── renderer.py            # Primary Pygame GUI handler
└── main.py                    # Root CLI argparser 
```

---

## 🛠 How To Run The Project

### 1. Install Dependencies
```bash
pip3 install torch numpy pygame pyyaml tensorboard matplotlib pandas
```

### 2. Start Training
Leverages GPU bounds tracking parallel implementations securely tracking outputs dynamically!
```bash
python3 main.py --mode train
```

### 3. Open Dashboards
Graphically view the results across decoupled async outputs avoiding GPU frame drops externally.
```bash
# Terminal B
python3 -m visualization.dashboard
# Terminal C
tensorboard --logdir logs/train
```

### 4. Evaluate The Model
Evaluate the best-saved architecture weights seamlessly triggering live GUI tracking physically simulating bounds correctly!
```bash
python3 main.py --mode evaluate
```

### 5. Automated Hyperparameter Tuning
Auto-spawn sequential subsets recording dataframes sorting highest permutations systematically.
```bash
python3 main.py --mode tune
```

---

## 🔧 Configuration System

The RL algorithms leverage standardized configuration YAMLs keeping logic strictly abstracted away dynamically providing inputs efficiently handling updates.

- **`hyperparameters.yaml`**: Main RL configs: `learning_rate`, `gamma`, `k_epochs`, `eps_clip`, `num_envs`, and bounds for sequential training limits. Target tuning updates affect internal networks structurally correctly avoiding logic loops.
- **`tune.yaml`**: Grid-based YAML matrix inputting multi-variant list items evaluated efficiently across sequential tuning bursts dynamically recording outputs gracefully! 
- **`tuning_results.yaml`**: Output location generated by grid search detailing top configurations accurately! 

---

## 📈 Training Metrics and Monitoring

The system outputs highly customized telemetry streams internally providing exact statistics analyzing model divergence mapping progress bounds internally ensuring success factors securely. The Tensorboard records `Loss/Policy`, `Loss/Value`, `Reward/Episode`, `Metrics/CrashRate`, and `Metrics/DifficultyLevel`. Matplotlib visually renders moving-avg progress natively asynchronously! 

---

## 🧩 Development Workflow

This system utilized extremely rigorous unit-testing integration validation dynamically triggering logic systems using `tasks.md` logic boundaries recursively tracking component completeness thoroughly mapping system milestones symmetrically! 

Every iteration visually mapped logic requirements passing validations natively securing integration cleanly scaling advanced vectors ensuring components never failed iteratively!

---

## 🔮 Future Improvements

1. **Obstacle Detection:** Introducing internal track logic components recursively generating randomized blockades parsing external sensor arrays mapping arrays securely internally evaluating dynamic objects correctly generating safety models successfully!
2. **LiDAR Raycasting Array:** Expanding raycasts radially accurately calculating point cloud representations internally providing depth grids dynamically!
3. **Multi-Agent Simulation:** Creating competitive sub-process models tracing collision impacts simulating generalized racing natively scaling systems internally globally!

---

## 🎯 Technical Summary

The Neural Tesla system elegantly scales generic custom 2D Pygame mathematical physics natively parallelized into optimized Multi-GPU memory frameworks correctly applying dynamic Curriculum tracks parsing highly specialized PPO metrics natively logging tracking securely providing bleeding edge results cleanly structurally perfectly implemented handling advanced requirements iteratively flawlessly! 
