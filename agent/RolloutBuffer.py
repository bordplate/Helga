import random

import torch
import numpy as np

from threading import Lock


class RolloutBuffer:
    def __init__(self, owner, capacity, buffer_size=512, gamma=0.99, lambda_gae=1, device='cpu', cell_size=256):
        self.owner = owner
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.position = 0
        self.total = 0
        self.last_episode_start = 0
        self.lock = Lock()
        self.new_samples = 0
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.cached = [None] * self.capacity

        self.buffer_size = buffer_size

        self.ready = False

        self.discounted_reward = 0

        #self.hidden_state = torch.zeros((cell_size), dtype=torch.bfloat16, device='cpu')
        #self.cell_state = torch.zeros((cell_size), dtype=torch.bfloat16, device='cpu')

        self.device = device

    def compute_returns_and_advantages(self, last_value, done):
        last_gae_lam = 0
        # Set the value of the next state to zero if the episode ends
        # next_value = 0 if done else last_value
        next_value = last_value
        mask = 1.0 - float(done)  # Convert `done` to float and invert

        # Reverse iteration over your buffer to calculate advantages and returns
        for step in reversed(range(self.last_episode_start, self.total)):
            if step < self.total - 1:
                mask = 1.0 - self.buffer[step + 1][3].float()

            next_value = next_value * mask
            last_gae_lam = last_gae_lam * mask

            # Calculate the delta according to the Bellman equation
            delta = self.buffer[step][2] + self.gamma * next_value - self.buffer[step][5]
            # Update last_gae_lam using the delta and decay terms
            last_gae_lam = delta + self.gamma * self.lambda_gae * last_gae_lam
            # The return is the value of the state plus the estimated advantage
            _return = last_gae_lam + self.buffer[step][5]

            # Replace the original transition with the new one that includes advantage and return
            self.buffer[step] = self.buffer[step] + (last_gae_lam, _return)

            next_value = self.buffer[step][5]

        self.last_episode_start = self.total

    def add(self, state, actions, reward, done, logprob, state_value, hidden_state, cell_state):
        if self.ready:
            return

        if self.total >= self.buffer_size:
            self.compute_returns_and_advantages(state_value.to('cpu'), done)
            self.ready = True
            return

        state = state.to('cpu')
        actions = actions.to('cpu')
        reward = torch.tensor(reward, dtype=torch.bfloat16, device='cpu')
        done = torch.tensor(done, dtype=torch.bool, device='cpu')
        logprob = logprob.to('cpu')
        state_value = state_value.to('cpu')

        self.lock.acquire()

        self.buffer[self.position] = (state, actions, reward, done, logprob, state_value, None, None)
        # self.buffer[self.position] = (state, actions, reward, done, logprob, state_value, hidden_state.to('cpu').unsqueeze(dim=0), cell_state.to('cpu').unsqueeze(dim=0))

        self.position = (self.position + 1) % self.capacity
        self.lock.release()

        self.new_samples += 1

        self.total += 1

    def clear(self):
        self.lock.acquire()

        self.buffer = [None] * self.capacity
        self.position = 0
        self.last_episode_start = 0
        self.total = 0
        self.ready = False
        self.cached = [None] * self.capacity

        self.lock.release()

    def get_batches(self, batch_size):
        """
        Shuffle the buffer and return a batch of transitions of size batch_size.

        :returns: Tensor of states, actions, rewards, dones, logprobs, state_values, hidden_states, cell_states
        """
        num_samples = self.total

        # Make num samples a multiple of batch size
        num_samples = num_samples - (num_samples % batch_size)

        # List of indices into self.buffer at batch_size intervals
        indices = list(range(0, num_samples, batch_size))
        random.shuffle(indices)

        for i in indices:
            if self.cached[i] is not None:
                yield self.cached[i]
                continue

            batch = self.buffer[i:i+batch_size]

            if len(batch) == batch_size and None not in batch:
                processed = self._process_batch(batch)

                self.cached[i] = processed

                yield processed

    def _process_batch(self, batch):
        with torch.no_grad():
            (states, actions, rewards, dones, logprobs, state_values, hidden_states, cell_states, advantages, returns) \
                = zip(*batch)

            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)
            dones = torch.stack(dones)
            logprobs = torch.stack(logprobs)
            # hidden_states = torch.stack(hidden_states)
            # cell_states = torch.stack(cell_states)
            advantages = torch.stack(advantages)
            returns = torch.stack(returns)

            return [states, actions, rewards, dones, logprobs, advantages, returns]
            # return [states, actions, rewards, dones, logprobs, hidden_states, cell_states, advantages, returns]