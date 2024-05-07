import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

import torch.cuda as cuda

from ActorCritic import ActorCritic
from Buffer import Buffer

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()


class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):
        self.action_std = action_std_init
        self.gamma = gamma
        self.lambda_gae = 0.95
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # self.train_stream = cuda.Stream(priority=-1)

        self.device = device

        self.batch_size = 1024
        self.replay_buffers = []

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ], eps=1e-5)

        self.mse_loss = nn.MSELoss()

    def start_new_episode(self):
        self.policy.reset_lstm_states()
        # self.policy_old.reset_lstm_states()

    def load_policy_dict(self, policy):
        self.policy.load_state_dict(policy)
        # self.policy_old.load_state_dict(policy)

    # def set_action_std(self, new_action_std):
    #     self.action_std = new_action_std
    #     self.policy.set_action_std(new_action_std)
    #     # self.policy_old.set_action_std(new_action_std)
    #
    # def decay_action_std(self, action_std_decay_rate, min_action_std):
    #     self.action_std = self.action_std = self.action_std - action_std_decay_rate
    #     self.action_std = round(self.action_std, 4)
    #
    #     if self.action_std <= min_action_std:
    #         self.action_std = min_action_std
    #
    #     self.set_action_std(self.action_std)

    def choose_action(self, state):
        with torch.no_grad():
            obs = np.array([state])
            state_sequence = torch.tensor(obs, dtype=torch.float).to(device)

            self.policy.eval()

            action, action_logprob, state_value, mu, log_std = self.policy.act(state_sequence)

        return action, action_logprob, state_value, mu, log_std

    def learn(self, buffer: Buffer):
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
                 mus, log_stds, hidden_states, cell_states, advantages, returns) in buffer.get_batches(self.batch_size):

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy, _, _, kl_div = self.policy.evaluate(states, actions, None, None)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs.mean(dim=1) - old_logprobs.squeeze(dim=1).detach().mean(dim=1))

                # Normalizing the advantages
                # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = advantages.squeeze()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = self.mse_loss(returns.squeeze(), state_values.squeeze())

                policy_losses.append(actor_loss.item())
                value_losses.append(critic_loss.item())

                kl_coef = 0.01
                ent_coef = 0.01

                entropy_loss = dist_entropy.sum(dim=1).mean()

                entropy_losses.append(entropy_loss.item())

                loss = actor_loss + 0.5 * critic_loss # - ent_coef * entropy_loss) #/ (buffer.batch_size / self.batch_size)
                # loss = actor_loss.mean() + 0.5 * critic_loss - kl_coef * kl_div + ent_coef * entropy_loss
                # loss = loss / (2048 / self.batch_size)
                # loss = actor_loss + 0.0 * entropy_loss + critic_loss * 0.5
                # loss = loss / 1
                # loss = actor_loss + 0.01 * dist_entropy + critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())

                # Check that the gradients are being updated
                # for name, param in self.policy.actor.named_parameters():
                #     if param.grad is not None:
                #         print(name, param.grad.norm().item())

                # Clip the gradients
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                self.optimizer.step()

        # for name, param in self.policy.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.norm().item())
        buffer.clear()

        return (np.mean(losses) if len(losses) > 0 else 0,
                np.mean(policy_losses) if len(policy_losses) > 0 else 0,
                np.mean(value_losses) if len(value_losses) > 0 else 0,
                np.mean(entropy_losses) if len(entropy_losses) > 0 else 0)
