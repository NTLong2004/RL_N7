import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =============================
# Reward shaping functions
# =============================

    # Kiểm tra tất cả các quân của token đã về đích theo thứ tự 57, 56, 55, 54
def all_in_winner_rank(state, token):
    # Nếu state là 1 chiều (1 quân mỗi người)
    if isinstance(state[token], int):
        return state[token] in [57, 56, 55, 54]
    # Nếu state là mảng nhiều quân mỗi người
    token_tokens = state[token]
    required_ranks = [57, 56, 55, 54]
    return sorted(token_tokens, reverse=True)[:4] == required_ranks

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
        print("Killed a token!")
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
    def __init__(self, num_token=4):
        self.num_token = num_token
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

# =============================
# SARSA Agent
# =============================
class SARSAAgent:
    def __init__(self, num_states=58, num_actions=6, alpha=0.1, gamma=0.95, 
                 epsilon=0.5, min_epsilon=0.01, decay=0.995):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.num_actions = num_actions

    def choose_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(1, self.num_actions+1))
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            q_values = self.Q[state]
            valid_qs = [(a, q_values[a-1]) for a in valid_actions]
            return max(valid_qs, key=lambda x: x[1])[0]

    def update(self, s, a, r, s_next, a_next):
        predict = self.Q[s, a-1]
        target = r + self.gamma * self.Q[s_next, a_next-1]
        self.Q[s, a-1] += self.alpha * (target - predict)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)





# =============================
# DQN Agent
# =============================
class QNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=6):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.95, epsilon=0.5, min_epsilon=0.01, decay=0.995):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.num_actions = 6

    def choose_action(self, state, valid_actions=None):
        # if valid_actions is None:
        #     valid_actions = list(range(1, self.num_actions+1))
        # if random.uniform(0,1) < self.epsilon:
        #     return random.choice(valid_actions)
        # state_tensor = torch.FloatTensor([state])
        # q_values = self.model(state_tensor)
        # q_valid = [(a, q_values[0, a-1].item()) for a in valid_actions]
        # return max(q_valid, key=lambda x: x[1])[0]
        if valid_actions is None:
            valid_actions = list(range(1, self.num_actions+1))
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)

    # Đưa state thành tensor có batch dimension
        state_tensor = torch.FloatTensor([[state]])
        q_values = self.model(state_tensor).squeeze(0)  # Kích thước [num_actions]

        q_valid = [(a, q_values[a-1].item()) for a in valid_actions]
        return max(q_valid, key=lambda x: x[1])[0]


    def update(self, s, a, r, s_next, done):
        # Thêm batch dimension để đồng nhất
        s_tensor = torch.FloatTensor([[s]])
        s_next_tensor = torch.FloatTensor([[s_next]])

        q_values = self.model(s_tensor).squeeze(0)  # → [output_dim]
        q_value = q_values[a-1]

        with torch.no_grad():
            next_q_values = self.model(s_next_tensor).squeeze(0)
            best_next_action = torch.argmax(next_q_values).item()
            target = r + (0 if done else self.gamma * next_q_values[best_next_action].item())

        loss = self.loss_fn(q_value, torch.tensor(target, dtype=torch.float))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
# =============================
# MC Agent
# =============================
class MCAgent:
    def __init__(self,num_states=58,num_actions=6,gamma=0.9,epsilon=0.5,min_epsilon=0.01,decay=0.995):
        self.Q=np.zeros((num_states,num_actions))
        self.returns={(s,a):[] for s in range(num_states) for a in range(num_actions)}
        self.gamma=gamma; self.epsilon=epsilon; self.min_epsilon=min_epsilon; self.decay=decay
        self.num_actions=num_actions; self.episode=[]

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def choose_action(self,state,valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(1,self.num_actions+1))
        if random.uniform(0,1) < self.epsilon:
            action = random.choice(valid_actions)
        else:
            q_valid = [(a,self.Q[state,a-1]) for a in valid_actions]
            action = max(q_valid,key=lambda x:x[1])[0]
        print("MC said:", action)
        return action

    def store(self,s,a,r): self.episode.append((s,a,r))

    def update(self):
        G=0; visited=set()
        for (s,a,r) in reversed(self.episode):
            G = self.gamma*G + r
            key=(int(s),int(a))
            if key not in visited:
                if key not in self.returns:
                    self.returns[key]=[]
                self.returns[key].append(G)
                self.Q[int(s),int(a)-1]=np.mean(self.returns[key])
                visited.add(key)
        self.episode=[]

# =============================
# Training Loop (SARSA+ DQN+ MC)
# =============================
def train_mixed_agents(episodes=500):
    env=LudoEnv()
    agents=[SARSAAgent(),SARSAAgent(),DQNAgent(),MCAgent()]
    rewards_per_agent=[[] for _ in range(env.num_token)]
    wins=[0]*env.num_token

    for ep in range(episodes):
        state,current_token=env.reset()
        actions=[agents[p].choose_action(state[p]) for p in range(env.num_token)]
        done_flags=[False]*env.num_token; total_rewards=[0]*env.num_token; winner=None

        while not all(done_flags):
            token=env.current_token; s=state[token]; a=actions[token]
            new_state,reward,done,p=env.step(a); s_next=new_state[token]
            if isinstance(agents[token],SARSAAgent):
                a_next=agents[token].choose_action(s_next)
                agents[token].update(s,a,reward,s_next,a_next)
                actions[token]=a_next
            elif isinstance(agents[token],DQNAgent):
                agents[token].update(s,a,reward,s_next,done)
                actions[token]=agents[token].choose_action(s_next)
            else: # MC
                agents[token].store(s,a,reward)
                actions[token]=agents[token].choose_action(s_next)
            total_rewards[token]+=reward; done_flags[token]=done
            if done and winner is None: winner=token; wins[token]+=1
            state=new_state

        for p in range(env.num_token):
            if isinstance(agents[p],MCAgent): agents[p].update()
            rewards_per_agent[p].append(total_rewards[p])

    return rewards_per_agent,wins

# =============================
# Run experiment
# =============================
rewards,wins=train_mixed_agents(episodes=1)

for p in range(4):
    algo="SARSA" if p<2 else ("DQN" if p==2 else "MC")
    plt.plot(rewards[p],label=f"Agent {p+1} ({algo})")
plt.xlabel("Episode"); plt.ylabel("Reward")
plt.title("So sánh SARSA vs DQN vs Monte Carlo trong Ludo")
plt.legend(); plt.show()

for p in range(4):
    algo="SARSA" if p<2 else ("DQN" if p==2 else "MC")
    print(f"Agent {p+1} ({algo}) thắng {wins[p]} trận")