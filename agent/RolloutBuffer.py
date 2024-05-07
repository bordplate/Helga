import random

import torch
import numpy as np

from threading import Lock


class RolloutBuffer:
    def __init__(self, owner, capacity, batch_size=512, gamma=0.99, lambda_gae=1):
        self.owner = owner
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.position = 0
        self.total = 0
        self.lock = Lock()
        self.new_samples = 0
        self.gamma = gamma
        self.lambda_gae = lambda_gae

        self.batch_size = batch_size

        self.ready = False

        self.discounted_reward = 0

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def compute_returns_and_advantages(self, last_value, done):
        last_gae_lam = 0
        # Set the value of the next state to zero if the episode ends
        next_value = 0 if done else last_value
        next_non_terminal = 1.0 - float(done)  # Convert `done` to float and invert

        # Reverse iteration over your buffer to calculate advantages and returns
        for step in reversed(range(self.total)):
            if step < self.total - 1:
                next_non_terminal = 1.0 - self.buffer[step + 1][3].float()
                next_value = self.buffer[step + 1][5]

            # Calculate the delta according to the Bellman equation
            delta = self.buffer[step][2] + self.gamma * next_value * next_non_terminal - self.buffer[
                step][5]
            # Update last_gae_lam using the delta and decay terms
            last_gae_lam = delta + self.gamma * self.lambda_gae * next_non_terminal * last_gae_lam
            # The return is the value of the state plus the estimated advantage
            _return = last_gae_lam + self.buffer[step][5]

            # Replace the original transition with the new one that includes advantage and return
            self.buffer[step] = self.buffer[step] + (last_gae_lam, _return)

        self.ready = True  # Mark the buffer as ready for training

    def add(self, state, actions, reward, done, logprob, state_value, mu, log_std):
        if self.ready:
            return

        if self.total >= self.batch_size:
            self.compute_returns_and_advantages(state_value, done)
            return

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        self.lock.acquire()

        self.buffer[self.position] = (state, actions, reward, done, logprob, state_value, mu, log_std)

        self.position = (self.position + 1) % self.capacity
        self.lock.release()

        self.new_samples += 1

        self.total += 1

    def clear(self):
        self.lock.acquire()
        self.buffer = [None] * self.capacity
        self.position = 0
        self.total = 0
        self.ready = False
        self.lock.release()

    def get_batches(self, batch_size):
        """
        Shuffle the buffer and return a batch of transitions of size batch_size.

        :returns: Tensor of states, actions, rewards, dones, logprobs, state_values, hidden_states, cell_states
        """
        num_samples = min(self.total, 30000)

        # Make num samples a multiple of batch size
        num_samples = num_samples - (num_samples % batch_size)

        # List of indices into self.buffer at batch_size intervals
        indices = list(range(0, num_samples, batch_size))
        random.shuffle(indices)

        for i in indices:
            batch = self.buffer[i:i+batch_size]

            if len(batch) == batch_size and None not in batch:
                yield self._process_batch(batch)

    def _process_batch(self, batch):
        (states, actions, rewards, dones, logprobs, state_values, mus, log_stds, advantages, returns) \
            = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        dones = torch.stack(dones).to(self.device)
        logprobs = torch.stack(logprobs).to(self.device)
        state_values = torch.stack(state_values).to(self.device)
        mus = torch.stack(mus).to(self.device)
        log_stds = torch.stack(log_stds).to(self.device)
        advantages = torch.stack(advantages).to(self.device)
        returns = torch.stack(returns).to(self.device)

        return states, actions, rewards, dones, logprobs, state_values, mus, log_stds, advantages, returns
