from collections import defaultdict
import numpy as np
import random
import pickle

class MonteCarloAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(action_size))
        self.returns = defaultdict(list)
        self.episode = []

    def select_action(self, state):
        state_key = tuple(np.round(state, 2))
        if np.random.rand() < self.epsilon:
            return random.randint(1, len(self.Q[state_key]))
        return np.argmax(self.Q[state_key]) + 1

    def store_transition(self, state, action, reward):
        self.episode.append((state, action, reward))

    def update_policy(self):
        G = 0
        visited = set()
        for state, action, reward in reversed(self.episode):
            G = self.gamma * G + reward
            state_key = tuple(np.round(state, 2))
            if (state_key, action) not in visited:
                self.returns[(state_key, action)].append(G)
                self.Q[state_key][action - 1] = np.mean(self.returns[(state_key, action)])
                visited.add((state_key, action))
        self.episode = []

    # === Lưu và tải bằng pickle ===
    def save(self, path="montecarlo_agent.pth"):
        with open(path, "wb") as f:
            pickle.dump({"Q": dict(self.Q), "returns": dict(self.returns)}, f)

    def load(self, path="montecarlo_agent.pth"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.Q = defaultdict(lambda: np.zeros(6), data["Q"])
            self.returns = defaultdict(list, data["returns"])
    # === Monte Carlo Agent ===
    def store_experience(self, state, action, reward, next_state, done):
        # """Lưu dữ liệu cho 1 episode (Monte Carlo cập nhật sau khi kết thúc)."""
        self.episode.append((state, action, reward))
    
        if done:
            self.update_policy()
            self.episode = []  # Xóa dữ liệu episode sau khi cập nhật
