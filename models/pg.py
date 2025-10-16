import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

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
        return self.fc(state)


# === Lớp tác nhân Policy Gradient (có lưu trạng thái đầy đủ) ===
class PolicyGradientAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

        # Bộ nhớ tạm thời cho 1 episode
        self.log_probs = []
        self.rewards = []

        # Bộ nhớ kết quả huấn luyện dài hạn
        self.episode_rewards = []
        self.total_episodes = 0

    def select_action(self, state):
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

        # Tính loss
        loss = []
        for log_prob, Gt in zip(self.log_probs, returns):
            loss.append(-log_prob * Gt)
        loss = torch.cat(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Dọn bộ nhớ tạm cho episode
        self.log_probs = []
        self.rewards = []

        self.total_episodes += 1

    # === Hàm lưu toàn bộ kết quả và trạng thái vào file .pth ===
    def save(self, path="policy_gradient_full.pth"):
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "gamma": self.gamma,
            "episode_rewards": self.episode_rewards,
            "total_episodes": self.total_episodes
        }
        torch.save(checkpoint, path)
        print(f"💾 Đã lưu trạng thái agent vào '{path}'")

    # === Hàm tải lại từ file .pth ===
    def load(self, path="policy_gradient_full.pth"):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.gamma = checkpoint.get("gamma", 0.99)
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.total_episodes = checkpoint.get("total_episodes", 0)
        print(f"✅ Đã tải lại agent từ '{path}' | Episode: {self.total_episodes}")
    def store_experience(self, state, action, reward, next_state, done):
        # Lưu phần thưởng để huấn luyện sau mỗi episode.
        self.store_reward(reward)
