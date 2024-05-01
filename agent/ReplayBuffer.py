import torch
import numpy as np

from collections import namedtuple

from threading import Lock


Transition = namedtuple('Transition', ('state', 'action', 'reward',
                                       'done',  'logprob', 'state_value', 'hidden_state', 'cell_state'))
FullTransition = namedtuple('Transition', ('state', 'action', 'reward',
                                       'done', 'logprob', 'state_value', 'hidden_state', 'cell_state'))
TransitionMessage = namedtuple('TransitionMessage', ('transition', 'worker_name'))


class ReplayBuffer:
    def __init__(self, owner, capacity, alpha=0.6):
        self.owner = owner
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = [None] * capacity
        self.position = 0
        self.total = 0
        self.lock = Lock()
        self.new_samples = 0

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

    def clear(self):
        self.lock.acquire()
        self.buffer = [None] * self.capacity
        self.position = 0
        self.total = 0
        self.lock.release()

    def sample(self, batch_size):
        if not self.buffer:
            return [], [], [], [], [], [], [], []

        self.lock.acquire()
        samples = self.buffer[:batch_size]

        self.lock.release()

        batch = FullTransition(*zip(*samples))
        states, actions, rewards, dones, logprobs, state_values = map(lambda x: torch.stack(x).to(self.device), batch[:-2])

        self.new_samples = 0

        return (states, actions, rewards, dones, logprobs, state_values,
                *map(lambda x: torch.stack(x).squeeze(2).permute(1, 0, 2).contiguous(), batch[-2:]))
