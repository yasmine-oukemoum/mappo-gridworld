
from environment import GridWorld

env = GridWorld()

observations = env.reset()
print("Initial observations:", observations)

env.render()

# Agent 0 moves right; Agent 1 moves left
actions = [3, 2]

observations, rewards, done, info = env.step(actions)

print("Rewards:", rewards)
print("Episode finished:", done)

env.render()
