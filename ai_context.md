# AI Context File (AGENTS.md)

## Project Overview

This repository builds a reinforcement learning system where a **neural
network trained from scratch** learns to drive a simulated car.

The agent receives sensor observations and outputs steering and throttle
commands.

The system should be modular and designed for experimentation with
reinforcement learning algorithms.

------------------------------------------------------------------------

## Tech Stack

Language:

Python 3.12

Libraries:

numpy\
torch\
pygame\
gymnasium\
tensorboard

------------------------------------------------------------------------

## Coding Standards

Variable naming:

snake_case

Class naming:

PascalCase

Constants:

UPPER_CASE

------------------------------------------------------------------------

## Code Quality Requirements

All code must include:

type hints

docstrings

modular structure

clear separation between simulation and training logic

Avoid:

global variables

monolithic files

hardcoded constants

------------------------------------------------------------------------

## File Structure

rl_car_rl/

env/ car.py track.py physics.py sensors.py environment.py

agent/ neural_network.py policy.py

training/ trainer.py reward_function.py training_loop.py

visualization/ renderer.py

configs/ hyperparameters.yaml

tests/

main.py

------------------------------------------------------------------------

## Neural Network Requirements

Network must:

be implemented manually using PyTorch

be modular

be independent from training code

support saving and loading checkpoints

------------------------------------------------------------------------

## Training Requirements

Training system must support:

checkpoint saving

resume training

reward logging

evaluation runs

------------------------------------------------------------------------

## Development Roadmap

Phase 1

Implement physics and simulation.

Phase 2

Implement sensors and state observation.

Phase 3

Build environment API.

Phase 4

Implement neural network policy.

Phase 5

Integrate reinforcement learning algorithm.

Phase 6

Add visualization.

Phase 7

Add training metrics and optimization tools.
