
import numpy as np


class GridWorld:
    def __init__(self, size=5):
        self.size = size

        # Initial positions of the two agents
        self.starts = [(0, 0), (0, 4)]

        # Each agent has its own target
        self.targets = [(4, 4), (4, 0)]

        self.reset()

    def reset(self):
        self.positions = list(self.starts)
        self.done = [False, False]
        return self.get_observations()

    def get_observations(self):
        # Each agent observes its own position and target
        observations = []

        for i in range(2):
            x, y = self.positions[i]
            tx, ty = self.targets[i]

            obs = np.array(
                [x, y, tx, ty],
                dtype=np.float32
            ) / (self.size - 1)

            observations.append(obs)

        return observations

    def step(self, actions):
        # 0 = up, 1 = down, 2 = left, 3 = right, 4 = stay
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0),
        }

        rewards = []


        for i, action in enumerate(actions):
            if self.done[i]:
                rewards.append(0.0)
                continue

            x, y = self.positions[i]
            tx, ty = self.targets[i]

            old_distance = abs(x - tx) + abs(y - ty)

            dx, dy = moves[int(action)]

            nx = int(np.clip(x + dx, 0, self.size - 1))
            ny = int(np.clip(y + dy, 0, self.size - 1))

            self.positions[i] = (nx, ny)

            new_distance = abs(nx - tx) + abs(ny - ty)

            # Small penalty for each step, plus progress toward target
            reward = -0.1 + 0.5 * (old_distance - new_distance)

            if self.positions[i] == self.targets[i]:
                self.done[i] = True
                reward += 10.0

            rewards.append(reward)


        episode_done = all(self.done)

        return (
            self.get_observations(),
            rewards,
            episode_done,
            {"positions": self.positions.copy()},
        )

    def render(self):
        grid = [["." for _ in range(self.size)]
                for _ in range(self.size)]

        for i, (x, y) in enumerate(self.targets):
            grid[x][y] = f"T{i}"

        for i, (x, y) in enumerate(self.positions):
            grid[x][y] = f"A{i}"

        print("\n".join(" ".join(row) for row in grid))
        print()
