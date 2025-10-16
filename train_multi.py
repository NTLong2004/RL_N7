import numpy as np

from models.pg import PolicyGradientAgent
from models.dqn import DQNAgent
from models.sarsa import SARSAAgent
from models.mc import MonteCarloAgent

NUM_EPISODES = 1000
SAVE_INTERVAL = 100
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
        print(f"token {self.current_token} is taking action {action}")
        print(f"Old state: {old_state}")
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

def train_multi_agent():
    env = LudoEnv()
    state_size = env.state_size = len(env.state)
    action_size = env.action_size = 6  # Xúc xắc 1-6

    # Khởi tạo 4 tác nhân với thuật toán khác nhau
    agents = [
        PolicyGradientAgent(state_size, action_size),
        DQNAgent(state_size, action_size),
        SARSAAgent(state_size, action_size),
        MonteCarloAgent(state_size, action_size)
    ]

    win_counts = np.zeros(env.num_tokens)
    best_rewards = [-float('inf')] * env.num_tokens

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        done = False
        total_rewards = [0] * env.num_tokens

        while not done:
            token = env.current_token
            agent = agents[token]

            # Use full environment state normalized for the agent
            normalized_state = np.array(state) / env.goal
            action = agent.select_action(normalized_state)

            # Step the environment with the single action of the current token
            new_state, reward, done, moved_token = env.step(action)

            # Store experience for the agent that acted (use normalized states if agent expects them)
            try:
                agent.store_experience(normalized_state, action, reward, np.array(new_state) / env.goal, done)
            except Exception:
                # Fallback to raw states if agent's store_experience signature differs
                agent.store_experience(state, action, reward, new_state, done)

            total_rewards[token] += reward
            state = new_state

        # Cập nhật chính sách sau mỗi trận
        for i, agent in enumerate(agents):
            if hasattr(agent, "update_policy"):
                agent.update_policy()

            # Theo dõi kết quả
            if total_rewards[i] > best_rewards[i]:
                if hasattr(agent, "save"):
                    agent.save(f"agent_{i+1}_best.pth")
                best_rewards[i] = total_rewards[i]

        # Xác định người thắng
        winner = int(np.argmax(total_rewards))
        win_counts[winner] += 1

        # Log
        if episode % 10 == 0:
            eps = getattr(agents[2], "epsilon", 0.0)
            print(f"Episode {episode} | Wins: {win_counts} | ε: {eps:.3f}")

        if episode % SAVE_INTERVAL == 0:
            for i, agent in enumerate(agents):
                if hasattr(agent, "save"):
                    agent.save(f"agent_{i+1}_ep{episode}.pth")

    print("Training completed.")
    print("Final win counts:", win_counts)

if __name__ == "__main__":
    train_multi_agent()
