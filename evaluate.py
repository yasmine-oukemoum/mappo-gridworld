
import numpy as np
import torch

from environment import GridWorld
from models import Actor

env = GridWorld()
actors = [Actor(), Actor()]

checkpoint = torch.load(
    "mappo_gridworld.pt",
    map_location="cpu",
    weights_only=True
)

for i in range(2):
    actors[i].load_state_dict(checkpoint["actors"][i])

for i in range(2):
    obs = torch.tensor(
        env.reset()[i],
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = actors[i](obs)
        probs = torch.softmax(logits, dim=-1)

    print(f"\nAgent A{i}")
    print("Logits:", logits.numpy().round(3))
    print("Probabilities:", probs.numpy().round(3))
    print("Best action:", probs.argmax().item())
successes = 0
episode_lengths = []

for episode in range(100):
    observations = env.reset()

    for step in range(20):
        actions = []

        for i in range(2):
            obs = torch.tensor(
                observations[i], dtype=torch.float32
            )

            with torch.no_grad():
                logits = actors[i](obs)
                action = torch.argmax(logits).item()

            actions.append(action)

        observations, rewards, done, _ = env.step(actions)

        if done:
            successes += 1
            episode_lengths.append(step + 1)
            break

print(f"Success rate: {successes}%")

if episode_lengths:
    print(
        "Average steps on successful episodes:",
        np.mean(episode_lengths)
    )

for episode in range(3):
    observations = env.reset()

    print(f"\n--- Episode {episode + 1} ---")
    env.render()

    for step in range(20):
        actions = []

        for i in range(2):
            obs = torch.tensor(
                observations[i], dtype=torch.float32
            )

            with torch.no_grad():
                logits = actors[i](obs)
                action = torch.argmax(logits).item()

            actions.append(action)

        observations, rewards, done, info = env.step(actions)

        print(
            f"Step {step + 1}: "
            f"actions={actions}, "
            f"positions={info['positions']}, "
            f"rewards={rewards}, "
            f"done={done}"
        )

        if done:
            print("SUCCESS: Both agents reached their targets!")
            break

    env.render()