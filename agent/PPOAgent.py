import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

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
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.device = device

        self.batch_size = 512

        self.replay_buffers = []

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.mse_loss = nn.MSELoss()

    def start_new_episode(self):
        self.policy.reset_lstm_states()
        # self.policy_old.reset_lstm_states()

    def load_policy_dict(self, policy):
        self.policy.load_state_dict(policy)
        # self.policy_old.load_state_dict(policy)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        # self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)

        if self.action_std <= min_action_std:
            self.action_std = min_action_std

        self.set_action_std(self.action_std)

    def choose_action(self, state):
        with torch.no_grad():
            obs = np.array([state])
            state_sequence = torch.tensor(obs, dtype=torch.float).to(device)

            self.policy.eval()

            action, action_logprob, state_value, mu, log_std = self.policy.act(state_sequence)

        return action.detach().cpu().flatten(), action, action_logprob, state_value, mu, log_std

    def learn(self, buffer: Buffer):
        if buffer.total < self.batch_size:
            return 0

        self.policy.train()
        self.policy.actor.train()
        self.policy.critic.train()

        losses = []
        policy_losses = []
        value_losses = []

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for (states, actions, _rewards, dones, old_logprobs, old_state_values,
                 mus, log_stds, hidden_states, cell_states) in buffer.get_batches(self.batch_size):
                old_state_values = old_state_values.squeeze().detach()

                # Monte Carlo estimate of returns
                _rewards = _rewards.flip(dims=[0])  # Reversing the rewards
                dones = dones.clone().detach().to(device).flip(dims=[0])  # Reversing the dones

                # Compute returns efficiently using tensor operations
                discounted_rewards = torch.zeros_like(_rewards)
                discounted_reward = 0.0
                for i, (reward, is_terminal) in enumerate(zip(_rewards, dones)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + (self.gamma * discounted_reward)
                    discounted_rewards[i] = discounted_reward

                discounted_rewards = discounted_rewards.flip(dims=[0])  # Flip back the rewards to original order

                # Normalizing the rewards
                discounted_rewards = discounted_rewards.to(device)
                rewards = (discounted_rewards - discounted_rewards.mean()) / (
                            discounted_rewards.std() + 1e-7)

                # hidden_states = hidden_states.detach()
                # cell_states = cell_states.detach()

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy, _, _, kl_div = self.policy.evaluate(states, actions, hidden_states, cell_states, mus, log_stds)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Calculating the advantages
                # Parameters for GAE
                lambda_gae = 0.95  # GAE parameter for weighting

                # Initialize gae and discounted_rewards
                advantages = torch.zeros_like(rewards)
                gae = 0
                next_value = 0  # This will be used for the last time step, where there is no next state

                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_value = 0  # No next state if it's the last timestep
                    else:
                        next_value = old_state_values[t + 1]

                    delta = rewards[t] + (self.gamma * next_value * (1 - dones[t].float())) - old_state_values[t]
                    gae = delta + (self.gamma * lambda_gae * (1 - dones[t].float()) * gae)
                    advantages[t] = gae

                # Normalizing the advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # rewards = advantages + old_state_values

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2)
                critic_loss = self.mse_loss(state_values.squeeze(), rewards)

                policy_losses.append(actor_loss.mean().item())
                value_losses.append(critic_loss.item())

                kl_coef = 0.01
                ent_coef = 0.01

                loss = actor_loss.mean() + 0.5 * critic_loss - kl_coef * kl_div + ent_coef * dist_entropy
                # loss = loss / (2048 / self.batch_size)
                # loss = actor_loss + 0.0 * entropy_loss + critic_loss * 0.5
                # loss = loss / 1
                # loss = actor_loss + 0.01 * dist_entropy + critic_loss

                self.optimizer.zero_grad()
                loss.mean().backward()
                losses.append(loss.mean().item())

                # Clip the gradients
                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.1)

                self.optimizer.step()

        # for name, param in self.policy.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.norm().item())
        buffer.clear_before_read_position()

        return (np.mean(losses) if len(losses) > 0 else 0,
                np.mean(policy_losses) if len(policy_losses) > 0 else 0,
                np.mean(value_losses) if len(value_losses) > 0 else 0)
