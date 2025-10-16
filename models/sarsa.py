import numpy as np
import random
from collections import defaultdict
import pickle

class SARSAAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99,
                 epsilon=0.1, lam=0.9, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lam = lam

        self.Q = defaultdict(lambda: np.zeros(action_size))
        self.memory = []

    def _state_key(self, state):
        if isinstance(state, (list, tuple, np.ndarray)):
            return tuple(int(x) for x in np.asarray(state).ravel())
        return (int(state),)

    def select_action(self, state):
        key = self._state_key(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.Q[key]))

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_policy(self):
        if not self.memory:
            return

        E = defaultdict(lambda: np.zeros(self.action_size))
        for i in range(len(self.memory) - 1):
            state, action, reward, next_state, done = self.memory[i]
            next_action = self.select_action(next_state)
            s_key = self._state_key(state)
            ns_key = self._state_key(next_state)
            a = int(action)

            reward = np.clip(reward, -10, 10) / 10.0

            q_predict = self.Q[s_key][a]
            q_target = reward + (0 if done else self.gamma * self.Q[ns_key][next_action])
            delta = q_target - q_predict

            E[s_key][a] += 1

            for s in list(self.Q.keys()):
                self.Q[s] += self.alpha * delta * E[s]
                E[s] *= self.gamma * self.lam

        self.memory.clear()
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="sarsa_agent.pth"):
        with open(path, "wb") as f:
            pickle.dump(dict(self.Q), f)

    def load(self, path="sarsa_agent.pth"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.Q = defaultdict(lambda: np.zeros(self.action_size), data)
