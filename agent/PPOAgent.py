import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

from ActorCritic import ActorCritic
from ReplayBuffer import ReplayBuffer

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

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()

    def start_new_episode(self):
        self.policy.reset_lstm_states()
        self.policy_old.reset_lstm_states()

    def load_policy_dict(self, policy):
        self.policy.load_state_dict(policy)
        self.policy_old.load_state_dict(policy)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

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

            self.policy_old.eval()

            action, action_logprob, state_value = self.policy_old.act(state_sequence)

        return action.detach().cpu().flatten(), action, action_logprob, state_value

    def learn(self, replay_buffer):
        if replay_buffer.total < self.batch_size:
            return 0

        self.policy.train()

        total_samples = min(replay_buffer.total, 1500)

        (states, actions, _rewards, dones, logprobs, state_values,
         hidden_states, cell_states) = replay_buffer.sample(total_samples)

        replay_buffer.clear()

        # Monte Carlo estimate of returns
        rewards = _rewards.clone().detach().to(device)
        dones = dones.clone().detach().to(device)

        # Initialize the next_value to 0.0 for the calculation of terminal states
        next_value = 0.0

        # We iterate backwards through rewards to accumulate return
        for idx in reversed(range(len(rewards))):
            if dones[idx]:
                next_value = 0  # If done, there is no next state; we reset next_value
            rewards[idx] = rewards[idx] + self.gamma * next_value
            next_value = rewards[idx]  # Update next_value to the current reward

        # Normalizing the rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # Prevent division by zero

        hidden_states = hidden_states.detach()
        cell_states = cell_states.detach()

        losses = []

        self.optimizer.zero_grad()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # If the replay buffer has already filled up, break out of the loop
            # We'd rather train on new data than old data
            if replay_buffer.total > 1500:
                print("Breaking out of PPO training loop early to prioritize new data.")
                break

            logprobs, state_values, dist_entropy, _, _ = (
                self.policy.evaluate(states, actions, hidden_states, cell_states))

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - logprobs.detach())

            adv = rewards - state_values.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Surrogate loss
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values.squeeze(), rewards) - 0.01 * dist_entropy

            # Gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()

            losses.append(loss.mean().item())

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        return np.mean(losses)
