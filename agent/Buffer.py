import random

import torch
import numpy as np

from collections import namedtuple

from threading import Lock


Transition = namedtuple('Transition', ('state', 'action', 'reward',
                                       'done',  'logprob', 'state_value', 'hidden_state', 'cell_state'))
FullTransition = namedtuple('Transition', ('state', 'action', 'reward',
                                       'done', 'logprob', 'state_value', 'hidden_state', 'cell_state'))
TransitionMessage = namedtuple('TransitionMessage', ('transition', 'worker_name'))

class Buffer:
    def __init__(self, owner, capacity, alpha=0.6):
        self.owner = owner
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = [None] * capacity
        self.position = 0
        self.total = 0
        self.lock = Lock()
        self.new_samples = 0

        self.read_position = 0

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def add(self, state, actions, reward, done, logprob, state_value, last_hidden_state, last_cell_state):
        state = torch.tensor(state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        hidden_state = last_hidden_state
        cell_state = last_cell_state

        self.lock.acquire()

        self.buffer[self.position] = (state, actions, reward, done, logprob, state_value, hidden_state, cell_state)

        self.position = (self.position + 1) % self.capacity
        self.lock.release()

        self.new_samples += 1

        self.total += 1

    def lock_read_position(self):
        self.read_position = self.total

    def clear(self):
        self.lock.acquire()
        self.buffer = [None] * self.capacity
        self.position = 0
        self.total = 0
        self.lock.release()

    def clear_before_read_position(self):
        self.lock.acquire()
        self.total = self.total - self.read_position
        self.buffer = self.buffer[self.read_position:] + [None] * (self.capacity - self.total)
        self.position = 0
        self.lock.release()

    def get_batches(self, batch_size):
        """
        Shuffle the buffer and return a batch of transitions of size batch_size.

        :returns: Tensor of states, actions, rewards, dones, logprobs, state_values, hidden_states, cell_states
        """
        num_samples = min(self.read_position, 2048)
        self.read_position = num_samples

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
        states, actions, rewards, dones, logprobs, state_values, hidden_states, cell_states = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        dones = torch.stack(dones).to(self.device)
        logprobs = torch.stack(logprobs).to(self.device)
        state_values = torch.stack(state_values).to(self.device)

        hidden_states = torch.stack(hidden_states).squeeze(dim=-2).permute(1, 0, 2).to(self.device)
        cell_states = torch.stack(cell_states).squeeze(dim=-2).permute(1, 0, 2).to(self.device)

        # Make states contiguous
        hidden_states = hidden_states.contiguous()
        cell_states = cell_states.contiguous()

        return states, actions, rewards, dones, logprobs, state_values, hidden_states, cell_states
