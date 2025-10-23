# from collections import defaultdict
# import numpy as np
# import random
# import pickle

# class MonteCarloAgent:
#     def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.1):
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.Q = defaultdict(lambda: np.zeros(action_size))
#         self.returns = defaultdict(list)
#         self.episode = []

#     def select_action(self, state):
#         state_key = tuple(np.round(state, 2))
#         if np.random.rand() < self.epsilon:
#             return random.randint(1, len(self.Q[state_key]))
#         return np.argmax(self.Q[state_key]) + 1

#     def store_transition(self, state, action, reward):
#         self.episode.append((state, action, reward))

#     def update_policy(self):
#         G = 0
#         visited = set()
#         for state, action, reward in reversed(self.episode):
#             G = self.gamma * G + reward
#             state_key = tuple(np.round(state, 2))
#             if (state_key, action) not in visited:
#                 self.returns[(state_key, action)].append(G)
#                 self.Q[state_key][action - 1] = np.mean(self.returns[(state_key, action)])
#                 visited.add((state_key, action))
#         self.episode = []

#     # === Lưu và tải bằng pickle ===
#     def save(self, path="montecarlo_agent.pth"):
#         with open(path, "wb") as f:
#             pickle.dump({"Q": dict(self.Q), "returns": dict(self.returns)}, f)

#     def load(self, path="montecarlo_agent.pth"):
#         with open(path, "rb") as f:
#             data = pickle.load(f)
#             self.Q = defaultdict(lambda: np.zeros(6), data["Q"])
#             self.returns = defaultdict(list, data["returns"])
#     # === Monte Carlo Agent ===
#     def store_experience(self, state, action, reward, next_state, done):
#         # """Lưu dữ liệu cho 1 episode (Monte Carlo cập nhật sau khi kết thúc)."""
#         self.episode.append((state, action, reward))
    
#         if done:
#             self.update_policy()
#             self.episode = []  # Xóa dữ liệu episode sau khi cập nhật

from collections import defaultdict
import numpy as np
import random
import pickle

class MonteCarloAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.1):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size

        # Q-table và bộ đếm lượt
        self.Q = defaultdict(lambda: np.zeros(action_size))
        self.N = defaultdict(lambda: np.zeros(action_size))  # đếm số lần (s,a) xuất hiện
        self.episode = []  # lưu (state, action, reward)

    def _state_key(self, state):
        """Chuyển state (mảng) sang tuple để làm key."""
        if isinstance(state, (list, tuple, np.ndarray)):
            return tuple(int(x) for x in np.asarray(state).ravel())
        return (int(state),)

    def select_action(self, state):
        """Chọn hành động theo epsilon-greedy."""
        key = self._state_key(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.Q[key]))

    def store_experience(self, state, action, reward, next_state=None, done=False):
        """Lưu trải nghiệm của 1 episode (Monte Carlo chỉ cập nhật sau khi episode kết thúc)."""
        self.episode.append((state, action, reward))
        if done:
            self.update_policy()
            self.episode.clear()

    def update_policy(self):
        """Cập nhật Q-value dựa trên toàn bộ episode."""
        G = 0
        visited = set()
        # duyệt ngược để tính return G_t
        for state, action, reward in reversed(self.episode):
            key = self._state_key(state)
            G = self.gamma * G + reward

            # Tránh cập nhật lặp nếu (s,a) đã xuất hiện
            if (key, action) not in visited:
                visited.add((key, action))
                self.N[key][action] += 1
                # Trung bình lũy thừa
                alpha = 1 / self.N[key][action]
                self.Q[key][action] += alpha * (G - self.Q[key][action])

    # === Lưu và tải bằng pickle ===
    def save(self, path="montecarlo_agent.txt"):
        with open(path, "wb") as f:
            pickle.dump({
                "Q": dict(self.Q),
                "N": dict(self.N)
            }, f)

    def load(self, path="montecarlo_agent.pth"):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.Q = defaultdict(lambda: np.zeros(self.action_size), data["Q"])
            self.N = defaultdict(lambda: np.zeros(self.action_size), data["N"])