import torch
import torch.nn as nn
import numpy as np

from PPO.ActorCritic import ActorCritic
from RolloutBuffer import RolloutBuffer
from RE3.RandomEncoder import RandomEncoder

class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor=1e-4,
                 lr_critic=1e-4,
                 batch_size=1024 * 8,
                 mini_batch_size=1024,
                 gamma=0.99,
                 K_epochs=5,
                 eps_clip=0.2,
                 log_std=-3.0,
                 ent_coef=0.01,
                 cl_coeff=0.5,
                 max_grad_norm=0.5,
                 beta=0.1,
                 kl_threshold=0.1,
                 lambda_gae=0.95,
                 device='cpu'
                 ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef
        self.cl_coeff = cl_coeff
        self.max_grad_norm = max_grad_norm
        self.beta = beta
        self.kl_threshold = kl_threshold
        self.lambda_gae = lambda_gae

        self.device = device

        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_size
        self.replay_buffers = []

        self.random_encoder = RandomEncoder(state_dim).to(device)

        self.policy = ActorCritic(state_dim, action_dim, log_std).to(device)

        self.action_mask = torch.ones(action_dim).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ], eps=1e-5)

        self.mse_loss = nn.MSELoss()

    def start_new_episode(self):
        pass

    def load_policy_dict(self, policy):
        self.policy.load_state_dict(policy)

    def choose_action(self, state_sequence: torch.Tensor):
        with torch.no_grad():
            self.policy.eval()

            action, action_logprob, state_value = self.policy.act(state_sequence, self.action_mask)

        return action, action_logprob, state_value

    def learn(self):
        self.policy.train()
        self.policy.actor.train()
        self.policy.critic.train()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_approx_kl = 0

        count = 0

        # Optimize policy for K epochs
        for epoch in range(self.K_epochs):
            for (n, buffer) in enumerate(self.replay_buffers):
                for (states, actions, _rewards, dones, old_logprobs, old_state_values,
                        advantages, returns) in buffer.get_batches(self.mini_batch_size):

                    # Evaluating old actions and values
                    logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions, self.action_mask)
                    # source_y_t = self.random_encoder(states[:, -1, :].squeeze())

                    # intrinsic_rewards = self.random_encoder.compute_intrinsic_rewards(source_y_t[:, :, 0, 0], y_t.squeeze(), True)
                    # intrinsic_rewards = intrinsic_rewards.squeeze().to(device)

                    advantages = advantages.squeeze().detach()
                    returns = returns.squeeze().detach()
                    old_logprobs = old_logprobs.squeeze(dim=1).detach()

                    # advantages = advantages + self.beta * intrinsic_rewards
                    # returns = returns + self.beta * intrinsic_rewards

                    # Finding the ratio (pi_theta / pi_theta__old)
                    ratios = torch.exp(logprobs.sum(dim=1) - old_logprobs.sum(dim=1))
                    # ratios = torch.exp(logprobs - old_logprobs.squeeze(dim=1).detach())

                    # Normalizing the advantages
                    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

                    surr1 = advantages * ratios
                    surr2 = advantages * torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)

                    actor_loss = -(torch.min(surr1, surr2)).mean()
                    critic_loss = self.mse_loss(state_values.squeeze(), returns.squeeze())

                    total_policy_loss += actor_loss.item()
                    total_value_loss += critic_loss.item()

                    entropy_loss = dist_entropy.sum(dim=1).mean()
                    total_entropy_loss += entropy_loss.item()

                    approx_kl = -((logprobs - old_logprobs).mean())
                    total_approx_kl += abs(approx_kl.item())
                    count += 1

                    loss = actor_loss - self.ent_coef * entropy_loss + self.cl_coeff * critic_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    total_loss += loss.item()

                    # Clip the gradients
                    nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    # Stop early if kl_divergence is above threshold
                    if total_approx_kl / count > self.kl_threshold:
                        print(f"Stopping early in {n}:{epoch}:{count} due to high KL divergence: {total_approx_kl / count }")
                        break

                # Stop early if kl_divergence is above threshold
                if total_approx_kl / count > self.kl_threshold:
                    break

            # Stop early if kl_divergence is above threshold
            if total_approx_kl / count > self.kl_threshold:
                break

        torch.cuda.empty_cache()

        # Clear the buffers
        for buffer in self.replay_buffers:
            buffer.clear()

        return (total_loss / count if count > 0 else 0,
                total_policy_loss / count if count > 0 else 0,
                total_value_loss / count if count > 0 else 0,
                total_entropy_loss / count if count > 0 else 0,
                total_approx_kl / count if count > 0 else 0)
