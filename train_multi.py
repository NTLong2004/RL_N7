import numpy as np

from models.pg import PolicyGradientAgent
from models.dqn import DQNAgent
from models.sarsa import SARSAAgent
from models.mc import MonteCarloAgent

NUM_EPISODES = 1000
SAVE_INTERVAL = 100
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

    win_counts = np.zeros(env.num_players)
    best_rewards = [-float('inf')] * env.num_players

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        done = False
        total_rewards = [0] * env.num_players

        while not done:
            player = env.current_player
            agent = agents[player]

            # Use full environment state normalized for the agent
            normalized_state = np.array(state) / env.goal
            action = agent.select_action(normalized_state)

            # Step the environment with the single action of the current player
            new_state, reward, done, moved_player = env.step(action)

            # Store experience for the agent that acted (use normalized states if agent expects them)
            try:
                agent.store_experience(normalized_state, action, reward, np.array(new_state) / env.goal, done)
            except Exception:
                # Fallback to raw states if agent's store_experience signature differs
                agent.store_experience(state, action, reward, new_state, done)

            total_rewards[player] += reward
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
