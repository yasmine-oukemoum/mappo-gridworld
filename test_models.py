
import torch

from environment import GridWorld
from models import Actor, Critic

env = GridWorld()
observations = env.reset()

# Two independent actors
actors = [Actor(), Actor()]

# One centralized critic
critic = Critic()

obs_tensors = [
    torch.tensor(obs, dtype=torch.float32)
    for obs in observations
]

actions = []

for i in range(2):
    action, log_prob = actors[i].get_action(obs_tensors[i])
    actions.append(action)

global_state = torch.cat(obs_tensors)

value = critic(global_state)

print("Actions:", actions)
print("Centralized state:", global_state)
print("Estimated state value:", value.item())

next_obs, rewards, done, info = env.step(actions)

print("Rewards:", rewards)
env.render()
