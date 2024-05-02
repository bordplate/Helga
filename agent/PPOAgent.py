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

        self.batch_size = 256

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

            action, action_logprob, state_value = self.policy.act(state_sequence)

        return action.detach().cpu().flatten(), action, action_logprob, state_value

    def learn(self, buffer: Buffer):
        if buffer.total < self.batch_size:
            return 0

        torch.autograd.set_detect_anomaly(True)

        self.policy.train()
        self.policy.actor.train()
        self.policy.critic.train()

        losses = []

        self.optimizer.zero_grad()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for (states, actions, _rewards, dones, old_logprobs, old_state_values,
                 hidden_states, cell_states) in buffer.get_batches(self.batch_size):
                # Monte Carlo estimate of returns
                rewards = []
                dones = dones.clone().detach().to(device)

                # Initialize the next_value to 0.0 for the calculation of terminal states
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(_rewards), reversed(dones)):
                    if is_terminal:
                        discounted_reward = 0
                    discounted_reward = reward + (self.gamma * discounted_reward)

                    rewards.insert(0, discounted_reward)

                # Normalizing the rewards
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # Prevent division by zero

                # hidden_states = hidden_states.detach()
                # cell_states = cell_states.detach()

                # Evaluating old actions and values
                logprobs, state_values, dist_entropy, _, _ = self.policy.evaluate(states, actions, hidden_states, cell_states)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Calculating the advantages
                advantages = rewards - state_values.detach()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                actor_loss = -torch.min(surr1, surr2)
                critic_loss = self.mse_loss(state_values.squeeze(), rewards)

                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy
                # loss = actor_loss + 0.01 * dist_entropy + critic_loss

                # Take gradient step
                loss.mean().backward()
                losses.append(loss.mean().item())

            # Clip the gradients
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)

            self.optimizer.step()
            self.optimizer.zero_grad()

        # for name, param in self.policy.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad.norm().item())
        buffer.clear_before_read_position()

        return np.mean(losses) if len(losses) > 0 else 0
