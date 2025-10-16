from models.pg import PolicyGradientAgent
from models.dqn import DQNAgent
from models.sarsa import SARSAAgent
from models.mc import MonteCarloAgent
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
def train_all_agents(num_episodes=500):
    env = LudoEnv()

    # === Khởi tạo 4 tác nhân ===
    state_size = env.state_size = 4  # 4 quân mỗi người
    action_size = env.action_size  = 6  # Xúc xắc 6 mặt

    agents = {
        "PolicyGradient": PolicyGradientAgent(state_size, action_size),
        "DQN": DQNAgent(state_size, action_size),
        "SARSA": SARSAAgent(state_size, action_size),
        "MonteCarlo": MonteCarloAgent(state_size, action_size)
    }

    # === Huấn luyện lần lượt ===
    for name in ["SARSA", "MonteCarlo"]:
        agent = agents[name]
        print(f"\n🚀 Đang huấn luyện tác nhân {name}...\n")
        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0

            done = False
            step = 0
            while not done and step < 500:
                normalized_state = np.array(state) / env.goal
                action = agent.select_action(normalized_state)
                new_state, reward, done, _ = env.step(action)

                agent.store_experience(state, action, reward, new_state, done)
                total_reward += reward
                state = new_state
                step += 1

            # Cập nhật sau mỗi episode (tùy thuật toán)
            agent.update_policy()
            episode_rewards.append(total_reward)

            if (episode + 1) % 50 == 0:
                avg_r = np.mean(episode_rewards[-50:])
                print(f"{name} | Episode {episode+1}/{num_episodes} | Avg Reward: {avg_r:.2f}")

        # Lưu model sau huấn luyện
        save_path = f"models/{name.lower()}_agent.pth"
        agent.save(save_path)
        print(f"✅ Đã lưu mô hình {name} tại {save_path}")


if __name__ == "__main__":
    train_all_agents(num_episodes=1000)