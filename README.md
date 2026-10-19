# MAPPO GridWorld — Multi-Agent Reinforcement Learning

A simple implementation of **Multi-Agent Proximal Policy Optimization (MAPPO)** using PyTorch. Two agents learn to navigate a grid environment and reach their individual targets through reinforcement learning.

## Project Overview

This project explores how multiple agents can learn coordinated behavior through independent policies and centralized value estimation.

Each agent:
- Observes its position and its assigned target.
- Chooses an action: up, down, left, right, or stay.
- Receives a reward based on its progress toward the target.
- Learns a policy through PPO updates and Generalized Advantage Estimation (GAE).

The implementation uses **decentralized actors and centralized critics**, a common architecture in multi-agent reinforcement learning.

## Architecture

```text
                 GRIDWORLD ENVIRONMENT
              (2 agents, 2 individual targets)
                           |
                           v
                  Agent Observations
                 [x, y, target_x, target_y]
                           |
                +----------+----------+
                |                     |
                v                     v
           +----------+          +----------+
           | Actor 0  |          | Actor 1  |
           | Policy π₀|          | Policy π₁|
           +----------+          +----------+
                |                     |
                v                     v
           Action 0              Action 1
                |                     |
                +----------+----------+
                           |
                           v
                  Environment Step
                           |
                 Rewards and New States
                           |
              +------------+------------+
              |                         |
              v                         v
       +-------------+           +-------------+
       |  Critic 0   |           |  Critic 1   |
       | Value V₀(s) |           | Value V₁(s) |
       +-------------+           +-------------+
              |                         |
              +------------+------------+
                           |
                           v
                 PPO + GAE Updates
                           |
                           v
                  Updated Networks
```

### Actor–Critic Design

**Actors (decentralized policies)**

Each actor receives its agent's local observation and outputs a probability distribution over five possible actions. The selected action is sampled from this distribution during training.

**Critics (centralized value estimation)**

Each critic receives the global state containing the positions of both agents and estimates the expected future return for its corresponding agent.

This separation allows actors to make decisions from local observations while critics use global information during training.

## Environment

The environment is a 5 × 5 grid with two agents and two fixed targets.

| Agent | Start | Target |
|---|---|---|
| Agent 0 | (0, 0) | (4, 4) |
| Agent 1 | (0, 4) | (4, 0) |

### Action Space

| Action | Movement |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |
| 4 | Stay |

### Observation Space

Each agent receives four normalized values:

`[x, y, target_x, target_y]`

The critic receives the combined state of both agents.

### Reward Design

The reward encourages agents to approach their targets:

- A small negative reward per step encourages shorter paths.
- A positive reward is given for reducing the Manhattan distance to the target.
- A large positive reward is given when an agent reaches its target.

## Reinforcement Learning Algorithm

The training process combines three key components:

- **PPO (Proximal Policy Optimization):** updates actor policies using a clipped objective to limit excessively large policy changes.
- **GAE (Generalized Advantage Estimation):** estimates how much better an action performed than expected, balancing bias and variance.
- **Critic learning:** trains value networks to estimate future returns and help calculate policy advantages.

The training loop collects trajectories, calculates returns and advantages, and updates the actor and critic networks over multiple optimization epochs.

## Project Structure

```text
mappo-gridworld/
├── environment.py       # GridWorld environment and reward function
├── models.py            # Actor and Critic neural networks
├── train.py             # MAPPO training loop
├── evaluate.py          # Evaluate trained actors
├── test_models.py       # Model tests
├── plot_results.py      # Plot training rewards
├── rewards.npy           # Recorded training rewards
├── mappo_gridworld.pt    # Saved model checkpoint
├── README.md             # Project documentation
└── .gitignore             # Files excluded from Git
```

## Installation

Requirements: Python 3.10+ recommended.

```bash
pip install numpy torch matplotlib
```

## Usage

### 1. Train the agents

```bash
python train.py
```

The script trains the agents and saves the model checkpoint and training rewards.

### 2. Evaluate the trained agents

```bash
python evaluate.py
```

This evaluates the learned policies over multiple episodes.

### 3. Plot training performance

```bash
python plot_results.py
```

## Results

In the current experiment:

- **Training episodes:** 1,500
- **Evaluation episodes:** 100
- **Evaluation success rate:** 100%
- **Average steps per successful episode:** 8

These results apply to the current fixed GridWorld setup. Further experiments across random seeds and environment configurations are needed to assess robustness.

## Future Improvements

- Compare MAPPO with random and shortest-path baseline policies.
- Evaluate performance across multiple random seeds.
- Randomize agent starting positions and target locations.
- Experiment with different reward functions and grid sizes.
- Add visualizations of agent trajectories.
- Investigate agent collisions and cooperative task settings.

## Technologies

Python · PyTorch · NumPy · Matplotlib · Reinforcement Learning · Multi-Agent Systems
