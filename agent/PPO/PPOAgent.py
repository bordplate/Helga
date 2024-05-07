import torch
import torch.nn as nn
import numpy as np

from PPO.ActorCritic import ActorCritic
from RolloutBuffer import RolloutBuffer

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


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
                 ent_coef=0.01
                 ):
        self.gamma = gamma
        self.lambda_gae = 0.95
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.ent_coef = ent_coef

        self.device = device

        self.mini_batch_size = mini_batch_size
        self.batch_size = batch_size
        self.replay_buffers = []

        self.policy = ActorCritic(state_dim, action_dim, log_std).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ], eps=1e-5)

        self.mse_loss = nn.MSELoss()

    def start_new_episode(self):
        self.policy.reset_lstm_states()

    def load_policy_dict(self, policy):
        self.policy.load_state_dict(policy)

    def choose_action(self, state):
        with torch.no_grad():
            obs = np.array([state])
            state_sequence = torch.tensor(obs, dtype=torch.float).to(device)

            self.policy.eval()

            action, action_logprob, state_value, mu, log_std = self.policy.act(state_sequence)

        return action, action_logprob, state_value, mu, log_std

    def learn(self, buffer: RolloutBuffer):
        if buffer.total < self.batch_size:
            return 0

        self.policy.train()
        self.policy.actor.train()
        self.policy.critic.train()

        losses = []
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for (states, actions, _rewards, dones, old_logprobs, old_state_values,
                 mus, log_stds, advantages, returns) in buffer.get_batches(self.mini_batch_size):

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy, _, _, kl_div = self.policy.evaluate(states, actions, None, None)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs.mean(dim=1) - old_logprobs.squeeze(dim=1).detach().mean(dim=1))

                advantages = advantages.squeeze()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(returns.squeeze(), state_values.squeeze())

                policy_losses.append(actor_loss.item())
                value_losses.append(critic_loss.item())

                entropy_loss = dist_entropy.sum(dim=1).mean()

                entropy_losses.append(entropy_loss.item())

                loss = actor_loss + 0.5 * critic_loss  # - self.ent_coef * entropy_loss) #/ (buffer.batch_size / self.batch_size)

                self.optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())

                # Clip the gradients
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                self.optimizer.step()

        buffer.clear()

        return (np.mean(losses) if len(losses) > 0 else 0,
                np.mean(policy_losses) if len(policy_losses) > 0 else 0,
                np.mean(value_losses) if len(value_losses) > 0 else 0,
                np.mean(entropy_losses) if len(entropy_losses) > 0 else 0)
