import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# =============================
# Reward shaping functions
# =============================

    # Kiểm tra tất cả các quân của player đã về đích theo thứ tự 57, 56, 55, 54
def all_in_winner_rank(state, player):
    # Nếu state là 1 chiều (1 quân mỗi người)
    if isinstance(state[player], (int, np.integer)):
        return state[player] in [57, 56, 55, 54]
    # Nếu state là mảng nhiều quân mỗi người
    player_tokens = state[player]
    required_ranks = [57, 56, 55, 54]
    if hasattr(player_tokens, '__iter__'):
        return sorted(player_tokens, reverse=True)[:4] == required_ranks
    else:
        return player_tokens in required_ranks

def token_reached_winner_rank(state, player, action):
    return state[player] == 57 and (state[player] - action) < 57

def token_on_winner_path(state, player, action):
    return 51 <= state[player] <= 56

def token_reached_safe_spot(state, player, action):
    safe_spots = [1, 9, 14, 22, 27, 35, 40, 48]
    return state[player] in safe_spots

def token_left_home(old_state, new_state, player, action):
    return old_state[player] == 0 and new_state[player] != 0 and action == 6

def token_moved_forward(old_state, new_state, player, action):
    return new_state[player] > old_state[player]


def check_player_killed(old_state, new_state, current_player):
    safe_spots = [1, 9, 14, 22, 27, 35, 40, 48]
    killed = False

    new_pos = new_state[current_player]

    # Chỉ kiểm tra nếu người chơi thật sự di chuyển
    if new_pos != old_state[current_player]:
        for other in range(len(old_state)):
            if other != current_player:
                # Nếu vị trí mới trùng với người khác, và ô đó không an toàn → giết
                if new_pos == new_state[other] and new_pos not in safe_spots:
                    killed = True
                    # Đưa quân đối thủ về nhà (0)
                    new_state[other] = 0
    return killed


def calculate_reward(old_state, new_state, action_taken, currentPlayer, playerKilled=False):
    reward = 0
    if all_in_winner_rank(new_state, currentPlayer):
        reward += 100
    elif token_reached_winner_rank(new_state, currentPlayer, action_taken):
        reward += 50
    if token_on_winner_path(new_state, currentPlayer, action_taken):
        reward += 25
    if playerKilled:
        print("Killed a player!")
        reward += 20
    if token_reached_safe_spot(new_state, currentPlayer, action_taken):
        reward += 10
    if token_left_home(old_state, new_state, currentPlayer, action_taken):
        reward += 5
    if token_moved_forward(old_state, new_state, currentPlayer, action_taken):
        reward += 1
    if old_state[currentPlayer] == new_state[currentPlayer]:
        reward -= 10
    return reward

# =============================
# Ludo Environment
# =============================
class LudoEnv:
    def __init__(self, num_players=4):
        self.num_players = num_players
        self.goal = 57
        self.reset()

    def reset(self):
        self.state = [0] * self.num_players
        self.finished_counts = [0] * self.num_players
        self.current_player = 0
        return self.state, self.current_player

    def step(self, action):
        old_state = self.state.copy()
        move = action
        player = self.current_player

        if self.state[player] + move <= self.goal:
            new_pos = self.state[player] + move
            if new_pos == self.goal:
                finish_slot = self.goal - self.finished_counts[player]
                if finish_slot < 0:
                    finish_slot = 0
                self.state[player] = finish_slot
                self.finished_counts[player] += 1
            else:
                self.state[player] = new_pos

        playerKilled = check_player_killed(old_state, self.state, player)
        reward = calculate_reward(old_state, self.state, move, player, playerKilled)

        # Điều kiện thắng: về đích đủ 4 quân đầu tiên
        done = self.finished_counts[player] >= 4
        self.current_player = (self.current_player + 1) % self.num_players

        return self.state, reward, done, player
# =======================================
# Cài Đặt các tác nhân SARSA, DQN, MC
# =======================================
class SARSAAgent:
    def __init__(self, num_actions, state_size):
        self.num_actions = num_actions
        self.state_size = state_size
        self.Q = np.zeros((state_size, num_actions))  # Bảng Q
        self.alpha = 0.1  # Tốc độ học
        self.gamma = 0.99  # Hệ số giảm giá
        self.epsilon = 1.0  # Tỷ lệ khám phá
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, state):
        # Chuyển state sang chỉ số hợp lệ
        if isinstance(state, list):
            state_idx = int(state[0])
        else:
            state_idx = int(state)
        # Kiểm tra phạm vi
        if state_idx < 0 or state_idx >= self.state_size:
            print(f"Invalid state index {state_idx} in act!")
            return random.randrange(self.num_actions)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        return np.argmax(self.Q[state_idx])

    def update(self, state, action, reward, next_state, next_action):
            # Đảm bảo state là một chỉ số hợp lệ cho bảng Q
            # Nếu state là list, lấy phần tử đầu tiên hoặc chuyển sang số nguyên hợp lệ
            if isinstance(state, list):
                state_idx = int(state[0])
            else:
                state_idx = int(state)
            if isinstance(next_state, list):
                next_state_idx = int(next_state[0])
            else:
                next_state_idx = int(next_state)

            # Kiểm tra phạm vi
            if state_idx >= self.state_size or next_state_idx >= self.state_size:
                print(f"Invalid state {state_idx} or next_state {next_state_idx}!")
                return

            predict = self.Q[state_idx, action]
            target = reward + self.gamma * self.Q[next_state_idx, next_action]
            self.Q[state_idx, action] += self.alpha * (target - predict)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DQNAgent:
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def __init__(self, num_actions, state_size):
        self.num_actions = num_actions
        self.state_size = state_size
        self.memory = []
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration factor
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # 24 units, ReLU activation
        model.add(Dense(24, activation='relu'))  # Another layer with ReLU activation
        model.add(Dense(self.num_actions, activation='linear'))  # Output layer with 'linear' activation
        # Update Adam optimizer initialization here:
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        # Chuyển state sang numpy array 2D nếu là list hoặc 1D
        state_arr = np.array(state)
        if state_arr.ndim == 1:
            state_arr = state_arr.reshape(1, -1)
        q_values = self.model.predict(state_arr)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            # Chuyển state và next_state sang numpy array 2D
            state_arr = np.array(state)
            if state_arr.ndim == 1:
                state_arr = state_arr.reshape(1, -1)
            next_state_arr = np.array(next_state)
            if next_state_arr.ndim == 1:
                next_state_arr = next_state_arr.reshape(1, -1)
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.model.predict(next_state_arr)[0])
            target_f = self.model.predict(state_arr)
            target_f[0][action] = target
            self.model.fit(state_arr, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MonteCarloAgent:
    def __init__(self, num_actions, state_size):
        self.num_actions = num_actions
        self.state_size = state_size
        self.Q = np.zeros((state_size, num_actions))
        self.N = np.zeros((state_size, num_actions))  # Số lần hành động được thực hiện
        self.gamma = 0.99  # Hệ số giảm giá

    def act(self, state):
        return random.randrange(self.num_actions)

    def update(self, episode):
        G = 0
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            self.N[state, action] += 1
            self.Q[state, action] += (G - self.Q[state, action]) / self.N[state, action]
class AgentManager:
    def __init__(self, num_actions, state_size):
        # Tạo các tác nhân (agents)
        self.agents = [
            SARSAAgent(num_actions, state_size),  # Tác nhân 1: SARSA
            SARSAAgent(num_actions, state_size),  # Tác nhân 2: SARSA
            DQNAgent(num_actions, state_size),    # Tác nhân 3: DQN
            MonteCarloAgent(num_actions, state_size)  # Tác nhân 4: Monte Carlo
        ]

    def get_action(self, agent_id, state):
        agent = self.agents[agent_id]
        return agent.act(state)

    def update(self, agent_id, state, action, reward, next_state, next_action=None):
        agent = self.agents[agent_id]
        if isinstance(agent, SARSAAgent):
            agent.update(state, action, reward, next_state, next_action)
        elif isinstance(agent, DQNAgent):
            agent.remember(state, action, reward, next_state, done=False)
            agent.replay()
        elif isinstance(agent, MonteCarloAgent):
            # Khi sử dụng Monte Carlo, ta chỉ gọi update khi kết thúc một episode
            pass

    def decay_epsilon(self):
        for agent in self.agents:
            if isinstance(agent, SARSAAgent) or isinstance(agent, DQNAgent):
                agent.decay_epsilon()
env = LudoEnv(num_players=4)  # Khởi tạo môi trường Ludo
agent_manager = AgentManager(num_actions=6, state_size=env.num_players)

# Khởi tạo danh sách lưu tổng reward cho từng agent
num_agents = 4
num_episodes = 10
agent_rewards = [[] for _ in range(num_agents)]  # mỗi agent một list

for episode in range(num_episodes):
    state, current_player = env.reset()
    done = False
    episode_history = []
    episode_agent_rewards = [0 for _ in range(num_agents)]
    winner = None

    while not done:
        action = agent_manager.get_action(current_player, state)
        next_state, reward, done, _ = env.step(action)
        next_action = agent_manager.get_action((current_player + 1) % 4, next_state)
        agent_manager.update(current_player, state, action, reward, next_state, next_action)

        episode_history.append((state, action, reward))
        episode_agent_rewards[current_player] += reward
        state = next_state
        # Nếu đã xong, người chơi hiện tại là người thắng
        if done:
            winner = current_player
        current_player = (current_player + 1) % 4

    agent_manager.update(3, state, action, reward, next_state)
    agent_manager.decay_epsilon()
    for i in range(num_agents):
        agent_rewards[i].append(episode_agent_rewards[i])

    # Lưu số ván thắng cho từng agent
    if episode == 0:
        agent_wins = [0 for _ in range(num_agents)]
    if winner is not None:
        agent_wins[winner] += 1
    print(f"Episode {episode} completed. Winner: Agent {winner+1 if winner is not None else 'Unknown'}")

# Vẽ biểu đồ reward của các agent
plt.figure(figsize=(10,6))
for i in range(num_agents):
    plt.plot(agent_rewards[i], label=f'Agent {i+1}')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward của các tác nhân qua các episode')
plt.legend()
plt.show()

# Hiển thị số ván thắng của từng agent
print("\nSố ván thắng của từng tác nhân:")
for i in range(num_agents):
    print(f"Agent {i+1}: {agent_wins[i]} ván thắng")
