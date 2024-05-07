import numpy as np


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.run_var = None

        self.n_rewards = 0
        self.mean_rewards = None
        self.run_var_rewards = None

    def update(self, x, reward=0.0):
        x = np.array(x)  # Ensure x is an array
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.run_var = np.zeros_like(x)

        self.n += 1
        old_mean = np.copy(self.mean)
        self.mean += (x - self.mean) / self.n
        self.run_var += (x - old_mean) * (x - self.mean)

        if reward is not None:
            if self.mean_rewards is None:
                self.mean_rewards = 0.0
                self.run_var_rewards = 0.0

            self.n_rewards += 1
            old_mean_rewards = self.mean_rewards
            self.mean_rewards += (reward - self.mean_rewards) / self.n_rewards
            self.run_var_rewards += (reward - old_mean_rewards) * (reward - self.mean_rewards)

    def variance(self):
        return self.run_var / self.n if self.n > 1 else np.zeros_like(self.run_var)

    def standard_deviation(self):
        return np.sqrt(self.variance())