
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment import GridWorld
from models import Actor, Critic


# Hyperparameters
NUM_AGENTS = 2
NUM_EPISODES = 1500
MAX_STEPS = 20

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
UPDATE_EPOCHS = 5
ENTROPY_COEF = 0.01

torch.manual_seed(42)
np.random.seed(42)


env = GridWorld()

# One actor per agent
actors = [Actor() for _ in range(NUM_AGENTS)]

# Centralized critics: each sees the global state (8 features),
# but estimates the value for its own agent.
critics = [Critic(state_dim=8) for _ in range(NUM_AGENTS)]

actor_optimizers = [
    torch.optim.Adam(actor.parameters(), lr=LR)
    for actor in actors
]

critic_optimizers = [
    torch.optim.Adam(critic.parameters(), lr=LR)
    for critic in critics
]


def get_state(observations):
    return torch.tensor(
        np.concatenate(observations),
        dtype=torch.float32
    )


def select_action(actor, observation):
    obs = torch.tensor(observation, dtype=torch.float32)

    with torch.no_grad():
        logits = actor(obs)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

    return action.item(), log_prob.item()


def compute_gae(rewards, values, dones):
    advantages = []
    gae = 0.0

    # values has one extra entry: the bootstrap value
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])

        delta = (
            rewards[t]
            + GAMMA * values[t + 1] * mask
            - values[t]
        )

        gae = (
            delta
            + GAMMA * GAE_LAMBDA * mask * gae
        )

        advantages.insert(0, gae)

    advantages = torch.tensor(
        advantages, dtype=torch.float32
    )

    returns = advantages + torch.tensor(
        values[:-1], dtype=torch.float32
    )

    return advantages, returns


def train():
    reward_history = []
    success_history = []

    for episode in range(NUM_EPISODES):
        observations = env.reset()

        # Separate trajectory for each agent
        agent_data = []

        for _ in range(NUM_AGENTS):
            agent_data.append({
                "obs": [],
                "actions": [],
                "log_probs": [],
                "states": [],
                "rewards": [],
                "dones": [],
            })

        episode_reward = 0.0

        for step in range(MAX_STEPS):
            state = get_state(observations)

            actions = []
            active_agents = []

            # Select actions for agents that have not finished
            for i in range(NUM_AGENTS):
                active = not env.done[i]
                active_agents.append(active)

                if active:
                    action, log_prob = select_action(
                        actors[i], observations[i]
                    )

                    data = agent_data[i]
                    data["obs"].append(observations[i].copy())
                    data["actions"].append(action)
                    data["log_probs"].append(log_prob)
                    data["states"].append(state.clone())

                    actions.append(action)
                else:
                    # The environment ignores actions of finished agents
                    actions.append(4)

            next_observations, rewards, episode_done, _ = env.step(
                actions
            )

            episode_reward += float(np.mean(rewards))

            # Store individual rewards and individual termination flags
            for i in range(NUM_AGENTS):
                if active_agents[i]:
                    agent_data[i]["rewards"].append(
                        float(rewards[i])
                    )
                    agent_data[i]["dones"].append(
                        bool(env.done[i])
                    )

            observations = next_observations

            if episode_done:
                break

        # Train each actor using its own trajectory and reward
        for i in range(NUM_AGENTS):
            data = agent_data[i]

            # Safety check
            if len(data["rewards"]) == 0:
                continue

            states = torch.stack(data["states"])

            obs_batch = torch.tensor(
                np.array(data["obs"]),
                dtype=torch.float32
            )

            action_batch = torch.tensor(
                data["actions"],
                dtype=torch.long
            )

            old_log_probs = torch.tensor(
                data["log_probs"],
                dtype=torch.float32
            )

            rewards = data["rewards"]
            dones = data["dones"]

            # Centralized critic values for this agent
            with torch.no_grad():
                values = critics[i](states).squeeze(-1).tolist()

                if dones[-1]:
                    next_value = 0.0
                else:
                    next_state = get_state(observations)
                    next_value = critics[i](next_state).item()

                values.append(next_value)

            advantages, returns = compute_gae(
                rewards, values, dones
            )

            # Normalize advantages when possible
            if len(advantages) > 1:
                advantages = (
                    advantages - advantages.mean()
                ) / (advantages.std(unbiased=False) + 1e-8)

            # Update this agent's centralized critic
            for _ in range(UPDATE_EPOCHS):
                predicted_values = critics[i](states).squeeze(-1)

                value_loss = nn.functional.mse_loss(
                    predicted_values, returns
                )

                critic_optimizers[i].zero_grad()
                value_loss.backward()

                nn.utils.clip_grad_norm_(
                    critics[i].parameters(), 0.5
                )

                critic_optimizers[i].step()

            # PPO update for this agent's actor
            for _ in range(UPDATE_EPOCHS):
                new_log_probs, entropy = actors[i].evaluate_actions(
                    obs_batch, action_batch
                )

                ratio = torch.exp(
                    new_log_probs - old_log_probs
                )

                surrogate1 = ratio * advantages

                surrogate2 = torch.clamp(
                    ratio,
                    1 - CLIP_EPS,
                    1 + CLIP_EPS
                ) * advantages

                policy_loss = -torch.min(
                    surrogate1, surrogate2
                ).mean()

                loss = (
                    policy_loss
                    - ENTROPY_COEF * entropy.mean()
                )

                actor_optimizers[i].zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    actors[i].parameters(), 0.5
                )

                actor_optimizers[i].step()

        reward_history.append(episode_reward)
        success_history.append(float(all(env.done)))

        if (episode + 1) % 100 == 0:
            mean_reward = np.mean(reward_history[-100:])
            success_rate = 100 * np.mean(success_history[-100:])

            print(
                f"Episode {episode + 1:4d} | "
                f"Mean reward: {mean_reward:.2f} | "
                f"Success rate (last 100): {success_rate:.1f}%"
            )

    torch.save(
        {
            "actors": [
                actor.state_dict() for actor in actors
            ],
            "critics": [
                critic.state_dict() for critic in critics
            ],
        },
        "mappo_gridworld.pt",
    )

    np.save("rewards.npy", np.array(reward_history))

    print("Training complete. Models saved.")


if __name__ == "__main__":
    train()