import torch as T
import numpy as np

from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from Network import DeepQNetwork


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.005, eps_dec=9e-6, sequence_length=5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.learn_length = 10

        self.action_space = [i for i in range(n_actions)]
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr=lr, feature_count=input_dims, hidden_dims=256, n_actions=n_actions)
        self.Q_target = DeepQNetwork(lr=lr, feature_count=input_dims, hidden_dims=256, n_actions=n_actions)
        self.Q_target.freeze()
        self.update_target_network()

        self.state_memory = np.zeros((self.mem_size, self.sequence_length, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.sequence_length, input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.keyframe_size = [0] * self.mem_size
        self.hidden_state_memory = [None] * self.mem_size
        self.cell_state_memory = [None] * self.mem_size

        self.hidden_state = None
        self.cell_state = None

        self.replay_buffer = PrioritizedReplayBuffer(self.mem_size)

    def start_new_episode(self):
        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
            self.hidden_state = None

        if self.cell_state is not None:
            self.cell_state = self.cell_state.detach()
            self.cell_state = None

        self.hidden_state = T.zeros(self.Q_eval.num_layers, 1, self.Q_eval.lstm_units).to(self.Q_eval.device)
        self.cell_state = T.zeros(self.Q_eval.num_layers, 1, self.Q_eval.lstm_units).to(self.Q_eval.device)

    def update_target_network(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def store_transition(self, state_sequence, action, reward, next_state_sequence, done):
        # FIXME: Storing the hidden and cell states for each state is wasteful.
        self.replay_buffer.add(state_sequence, action, reward, next_state_sequence, done,)

    def choose_action(self, observation_sequence):
        if np.random.random() > self.epsilon:
            obs = np.array([observation_sequence])
            state_sequence = T.tensor(obs, dtype=T.float).to(self.Q_eval.device)

            self.Q_eval.eval()
            with T.no_grad():
                if self.hidden_state is None or self.cell_state is None:
                    actions, (self.hidden_state, self.cell_state) = self.Q_eval(state_sequence)
                else:
                    actions, (self.hidden_state, self.cell_state) = self.Q_eval(state_sequence, hidden_state=self.hidden_state,
                                                                                cell_state=self.cell_state)

            actions = actions[0]
            action = T.argmax(actions).item()
        else:
            action = np.random.choice([0, 1, 2, 3])

        return action

    def learn(self, num_batches=1, terminal_learn=False, average_reward=0.0):
        batch_size = self.batch_size * num_batches

        if len(self.replay_buffer.buffer) < batch_size:
            return 0

        self.Q_eval.train()
        self.Q_eval.optimizer.zero_grad()

        (states, actions, rewards, next_states, dones, indices, weights,
         hidden_states, cell_states, next_hidden_states, next_cell_states) = self.replay_buffer.sample(batch_size, beta=0.4)

        # Forward pass for current and next state batches
        _actions, _ = self.Q_eval(states, hidden_state=hidden_states, cell_state=cell_states)
        q_eval = _actions.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        q_next, _ = self.Q_eval(next_states, hidden_state=next_hidden_states, cell_state=next_cell_states)
        q_target_next, _ = self.Q_target(next_states, hidden_state=next_hidden_states, cell_state=next_cell_states)

        max_next_actions = T.argmax(q_next, dim=1)
        max_q_next = q_target_next.gather(1, max_next_actions.unsqueeze(-1)).squeeze(-1)
        max_q_next[dones] = 0.0

        # Calculate the Q target and loss
        q_target = rewards + self.gamma * max_q_next

        td_error = abs(q_target - q_eval).detach().cpu().numpy()

        self.replay_buffer.update_priorities(indices, td_error + 1e-5)

        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()

        T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), max_norm=0.5)

        self.Q_eval.optimizer.step()

        if terminal_learn:
            self.Q_eval.scheduler.step(average_reward)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

        return loss.item()
