from agents.pg import PolicyGradientAgent
from agents.dqn import DQNAgent
from agents.sarsa import SARSAAgent
from agents.mc import MonteCarloAgent
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

   # Kiểm tra tất cả các quân của token đã về đích theo thứ tự 57, 56, 55, 54
def all_in_winner_rank(state, token):
    # Nếu state là 1 chiều (1 quân mỗi người)
    if isinstance(state[token], (int, np.integer)):
        return state[token] in [57, 56, 55, 54]
    # Nếu state là mảng nhiều quân mỗi người
    token_tokens = state[token]
    required_ranks = [57, 56, 55, 54]
    if hasattr(token_tokens, '__iter__'):
        return sorted(token_tokens, reverse=True)[:4] == required_ranks
    else:
        return token_tokens in required_ranks

def token_reached_winner_rank(state, token, action):
    return state[token] == 57 and (state[token] - action) < 57

def token_on_winner_path(state, token, action):
    return 51 <= state[token] <= 56

def token_reached_safe_spot(state, token, action):
    safe_spots = [1, 9, 14, 22, 27, 35, 40, 48]
    return state[token] in safe_spots

def token_left_home(old_state, new_state, token, action):
    return old_state[token] == 0 and new_state[token] != 0 and action == 6

def token_moved_forward(old_state, new_state, token, action):
    return new_state[token] > old_state[token]


def check_token_killed(old_state, new_state, current_token):
    safe_spots = [1, 9, 14, 22, 27, 35, 40, 48]
    killed = False

    new_pos = new_state[current_token]

    # Chỉ kiểm tra nếu người chơi thật sự di chuyển
    if new_pos != old_state[current_token]:
        for other in range(len(old_state)):
            if other != current_token:
                # Nếu vị trí mới trùng với người khác, và ô đó không an toàn → giết
                if new_pos == new_state[other] and new_pos not in safe_spots:
                    killed = True
                    # Đưa quân đối thủ về nhà (0)
                    new_state[other] = 0
    return killed


def calculate_reward(old_state, new_state, action_taken, currenttoken, tokenKilled=False):
    reward = 0
    if all_in_winner_rank(new_state, currenttoken):
        reward += 100
    elif token_reached_winner_rank(new_state, currenttoken, action_taken):
        reward += 50
    if token_on_winner_path(new_state, currenttoken, action_taken):
        reward += 25
    if tokenKilled:
        reward += 20
    if token_reached_safe_spot(new_state, currenttoken, action_taken):
        reward += 10
    if token_left_home(old_state, new_state, currenttoken, action_taken):
        reward += 5
    if token_moved_forward(old_state, new_state, currenttoken, action_taken):
        reward += 1
    if old_state[currenttoken] == new_state[currenttoken]:
        reward -= 10
    return reward

# =============================
# Ludo Environment
# =============================
class LudoEnv:
    def __init__(self, num_tokens=4):
        self.num_tokens = num_tokens
        self.goal = 57
        self.reset()

    def reset(self):
        self.state = [0] * self.num_tokens
        self.finished_counts = [0] * self.num_tokens
        self.current_token = 0
        return self.state, self.current_token

    def step(self, action):
        old_state = self.state.copy()
        move = action
        token = self.current_token

        if self.state[token] + move <= self.goal:
            new_pos = self.state[token] + move
            if new_pos == self.goal:
                finish_slot = self.goal - self.finished_counts[token]
                if finish_slot < 0:
                    finish_slot = 0
                self.state[token] = finish_slot
                self.finished_counts[token] += 1
            else:
                self.state[token] = new_pos

        tokenKilled = check_token_killed(old_state, self.state, token)
        reward = calculate_reward(old_state, self.state, move, token, tokenKilled)

        # Điều kiện thắng: về đích đủ 4 quân đầu tiên
        done = self.finished_counts[token] >= 4
        self.current_token = (self.current_token + 1) % self.num_tokens

        return self.state, reward, done, token
def train_all_agents(num_episodes=500):
    env = LudoEnv()

    # === Khởi tạo 4 tác nhân ===
    state_size = env.state_size = 4  # 4 quân mỗi người
    action_size = env.action_size  = 6  # Xúc xắc 6 mặt

    agents = {
        "PolicyGradient": PolicyGradientAgent(state_size, action_size),
        # "PolicyGradient": PolicyGradientAgent(state_size, action_size),
        # "PolicyGradient": PolicyGradientAgent(state_size, action_size),
        # "PolicyGradient": PolicyGradientAgent(state_size, action_size),
        "DQN": DQNAgent(state_size, action_size),
        "SARSA": SARSAAgent(state_size, action_size),
        "MonteCarlo": MonteCarloAgent(state_size, action_size)
    }

    # === Huấn luyện lần lượt ===
    for name, agent in agents.items():
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
        save_path = f"datatrain/{name.lower()}_agent.pth"
        agent.save(save_path)
        print(f"✅ Đã lưu mô hình {name} tại {save_path}")


if __name__ == "__main__":
    train_all_agents(num_episodes=1000)