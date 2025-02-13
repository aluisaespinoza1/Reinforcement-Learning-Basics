import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQL Agent
class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0015
        self.batch_size = 32
        
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f.detach())
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train the agent
env = gym.make('Blackjack-v1', natural=False, sab=False)
state_size = 3  # (player_sum, dealer_card, usable_ace)
action_size = env.action_space.n
agent = DQLAgent(state_size, action_size)

episodes = 5000
win_rates = []
for e in range(episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
    agent.replay()
    
    if e % 500 == 0:
        print(f"Episode {e}/{episodes}, Epsilon: {agent.epsilon:.4f}")

# Test the trained agent
test_episodes = 1000
wins = 0
losses = 0
draws = 0
for _ in range(test_episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    while not done:
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1
    win_rates.append(wins / (wins + losses + draws))

print(f"Results after {test_episodes} episodes:")
print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")

# Plot results
#plt.plot(win_rates)
#plt.xlabel('Games Played')
#plt.ylabel('Win Rate')
#plt.title('Win Rate Over Time')
#plt.show()

# Plot results
plt.figure(figsize=(8, 5))

# Win Rate Over Time
plt.subplot(1, 2, 1)
plt.plot(win_rates, label='Win Rate')
plt.xlabel('Games Played')
plt.ylabel('Win Rate')
plt.title('Win Rate Over Time')
plt.legend()

# Bar Chart of Final Results
plt.subplot(1, 2, 2)
bars = plt.bar(['Wins', 'Losses', 'Draws'], [wins, losses, draws], color=['green', 'red', 'gray'])
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.title('Final Results Distribution')

# Add labels to bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.tight_layout()
plt.show()