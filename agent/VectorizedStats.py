import torch


class VectorizedStats:
    def __init__(self, epsilon=1e-4, shape=(), device='cpu'):
        self.mean = torch.zeros(shape, dtype=torch.bfloat16, device=device)
        self.var = torch.ones(shape, dtype=torch.bfloat16, device=device)
        self.count = torch.tensor(epsilon, dtype=torch.bfloat16, device=device)

        self.device = device

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class VecNormalize:
    def __init__(self, epsilon=1e-4, shape=(), gamma=0.99, cliprew=10.0, device='cpu'):
        self.obs_stats = VectorizedStats(epsilon, shape, device=device)
        self.ret_stats = VectorizedStats(epsilon, (), device=device)
        self.ret = torch.zeros(1, dtype=torch.float32, device=device)
        self.gamma = gamma
        self.cliprew = cliprew

        self.device = device

    def normalize_observation(self, observation):
        self.obs_stats.update(observation)
        return self.obs_stats.normalize(observation)

    def normalize_reward(self, reward):
        self.ret = self.ret * self.gamma + reward
        self.ret_stats.update(self.ret.unsqueeze(0))
        normalized_reward = reward / torch.sqrt(self.ret_stats.var + 1e-8)
        return torch.clamp(normalized_reward, -self.cliprew, self.cliprew).squeeze().to('cpu').numpy()

    def get_mean_var_count(self):
        return (self.obs_stats.mean, self.obs_stats.var, self.obs_stats.count), (self.ret_stats.mean, self.ret_stats.var, self.ret_stats.count)

    def set_mean_var_count(self, obs_mean, obs_var, obs_count, ret_mean, ret_var, ret_count):
        self.obs_stats.mean = obs_mean
        self.obs_stats.var = obs_var
        self.obs_stats.count = obs_count
        self.ret_stats.mean = ret_mean
        self.ret_stats.var = ret_var
        self.ret_stats.count = ret_count


if __name__ == "__main__":
    vec_normalize = VecNormalize(shape=(3,))
    observation = torch.rand(5, 3, dtype=torch.bfloat16)
    normalized_observation = vec_normalize.normalize_observation(observation)
    print("Normalized Observation:\n", normalized_observation)
    reward = torch.tensor([1.0], dtype=torch.bfloat16)
    normalized_reward = vec_normalize.normalize_reward(reward)
    print("Normalized Reward:", normalized_reward)
    (obs_mean, obs_var, obs_count), (ret_mean, ret_var, ret_count) = vec_normalize.get_mean_var_count()
    print("Observation Mean:", obs_mean)
    print("Observation Variance:", obs_var)
    print("Observation Count:", obs_count)
    print("Return Mean:", ret_mean)
    print("Return Variance:", ret_var)
    print("Return Count:", ret_count)
