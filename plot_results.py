
import numpy as np
import matplotlib.pyplot as plt

rewards = np.load("rewards.npy")

window = 50
moving_average = np.convolve(
    rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 5))
plt.plot(rewards, alpha=0.25, label="Episode reward")
plt.plot(
    np.arange(window - 1, len(rewards)),
    moving_average,
    label="Moving average (50 episodes)"
)

plt.xlabel("Episode")
plt.ylabel("Team reward")
plt.title("MAPPO Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
