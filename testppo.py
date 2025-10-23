import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
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


def check_token_killed(old_state, new_state, current_token):
    """Compatibility wrapper: some files use 'token' naming, others use 'player'.
    Delegate to check_player_killed to avoid duplicate logic.
    """
    return check_player_killed(old_state, new_state, current_token)


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
    def __init__(self, num_token=4, num_players=None):
        """Initialize environment.

        Accepts either num_token (backwards-compatible) or num_players keyword.
        """
        # Support calling with num_players for compatibility
        if num_players is not None:
            num_token = num_players
        self.num_token = num_token
        # Also set num_players attribute for callers that expect it
        self.num_players = num_token
        self.goal = 57
        self.reset()

    def reset(self):
        self.state = [0] * self.num_token
        self.finished_counts = [0] * self.num_token
        self.current_token = 0
        return self.state, self.current_token

    def step(self, action):
        old_state = self.state.copy()
        move = action
        token = self.current_token

        if self.state[token] + move <= self.goal:
            new_pos = self.state[token] + move
            # If the token lands exactly on the goal (exact roll), we place
            # it into the appropriate winner slot for that token. The first
            # finished token will occupy 57, the next 56, then 55, etc.
            if new_pos == self.goal:
                finish_slot = self.goal - self.finished_counts[token]
                # safety: never place beyond goal lower bound
                if finish_slot < 0:
                    finish_slot = 0
                self.state[token] = finish_slot
                self.finished_counts[token] += 1
            else:
                self.state[token] = new_pos

        tokenKilled = check_token_killed(old_state, self.state, token)
        reward = calculate_reward(old_state, self.state, move, token, tokenKilled)

        done = self.state[token] == self.goal
        self.current_token = (self.current_token + 1) % self.num_token

        return self.state, reward, done, token

# === Mạng chính sách (Policy Network) ===
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        probs = self.fc(state)
        return probs


# === Lớp tác nhân Policy Gradient ===
class PolicyGradientAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        # Bộ nhớ lưu trữ 1 episode
        self.log_probs = []
        self.rewards = []

    def select_action(self, state):
        """Chọn hành động dựa trên phân phối xác suất của policy"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        """Cập nhật policy sau 1 episode"""
        # Tính tổng phần thưởng có chiết khấu Gt
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        # Chuẩn hóa để ổn định huấn luyện
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = []
        for log_prob, Gt in zip(self.log_probs, returns):
            loss.append(-log_prob * Gt)
        loss = torch.cat(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Dọn bộ nhớ episode
        self.log_probs = []
        self.rewards = []

    def save(self, path="policy_agent.pth"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path="policy_agent.pth"):
        self.policy.load_state_dict(torch.load(path))
# =============================
# Hàm huấn luyện Policy Gradient
# =============================

def train_policy_gradient(env, agent, num_episodes=1000, max_steps=500):
    episode_rewards = []

    for episode in range(num_episodes):
        state, current_player = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Chuẩn hóa trạng thái đầu vào cho agent
            # (ở đây state là list vị trí của 4 người chơi)
            normalized_state = np.array(state) / env.goal

            # Agent chọn hành động (bước đi 1-6)
            action = agent.select_action(normalized_state)

            # Thực hiện bước đi trong môi trường
            new_state, reward, done, player = env.step(action)

            # Lưu reward để cập nhật sau
            agent.store_reward(reward)
            total_reward += reward

            # Nếu người chơi thắng → kết thúc episode
            if done:
                break

            state = new_state

        # Cập nhật policy sau 1 episode
        agent.update_policy()
        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"🎯 Episode {episode+1}/{num_episodes} | Avg Reward (last 50): {avg_reward:.2f}")

    print("✅ Huấn luyện hoàn tất!")
    return episode_rewards
# =============================
env = LudoEnv(num_players=4)
agent = PolicyGradientAgent(state_size=4, action_size=6, lr=1e-3, gamma=0.99)

rewards = train_policy_gradient(env, agent, num_episodes=100)

# Lưu model
agent.save("ludo_policy_gradient.pth")
