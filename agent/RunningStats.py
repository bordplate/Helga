import numpy as np
import torch


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.run_var = None

    def update(self, x):
        x = np.array(x)  # Ensure x is an array
        if self.mean is None:
            self.mean = np.zeros_like(x)
            self.run_var = np.zeros_like(x)

        self.n += 1
        old_mean = np.copy(self.mean)
        self.mean += (x - self.mean) / self.n
        self.run_var += (x - old_mean) * (x - self.mean)

    def variance(self):
        return self.run_var / self.n if self.n > 1 else np.zeros_like(self.run_var)

    def standard_deviation(self):
        return np.sqrt(self.variance()) + 1e-7


class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        x = torch.tensor(x)

        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)

def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
    ):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta * delta * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count