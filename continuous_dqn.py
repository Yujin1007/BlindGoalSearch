import numpy as np
import random
import matplotlib.pyplot as plt

from env.ContinuousGoalEnv import Continuous2DGridWorld

import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output bounded between [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state)
        return action, value

# Hyperparameters
state_dim = 4  # [position_x, position_y, goal_x, goal_y]
action_dim = 2  # [dx, dy]
max_episodes = 1000
max_steps = 100
gamma = 0.99
epsilon_clip = 0.2
lr = 0.001
update_steps = 64
ppo_epochs = 10

# Initialize environment and model
env = ContinuousGridWorld()
model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Experience collection
def collect_trajectories(env, model, rollout_size=2000):
    states, actions, rewards, dones, log_probs = [], [], [], [], []
    state = env.reset()

    for _ in range(rollout_size):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_mean, _ = model(state_tensor)
        action = action_mean.detach().numpy()*0.1 + np.random.normal(0, 0.01, size=action_dim)  # Add noise for exploration
        action = np.clip(action, -0.1, 0.1)

        next_state, reward, done = env.step(action)
        log_prob = -0.5 * ((action - action_mean.detach().numpy()) ** 2).sum()  # Log probability for Gaussian

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)

        state = next_state
        if done:
            state = env.reset()
    # print(actions)
    return np.array(states), np.array(actions), np.array(rewards), np.array(dones), np.array(log_probs)

# PPO Update
def ppo_update(model, optimizer, states, actions, rewards, dones, log_probs, old_values):
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    returns = compute_returns(rewards, dones, gamma)

    for _ in range(ppo_epochs):
        action_means, values = model(states_tensor)
        advantages = returns - values.detach().numpy()

        # Compute ratios
        new_log_probs = -0.5 * ((actions_tensor - action_means) ** 2).sum(dim=1)
        ratios = torch.exp(new_log_probs - torch.tensor(log_probs, dtype=torch.float32))

        # Policy loss
        surr1 = ratios * torch.tensor(advantages, dtype=torch.float32)
        surr2 = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * torch.tensor(advantages, dtype=torch.float32)
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), torch.tensor(returns, dtype=torch.float32))

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Compute discounted returns
def compute_returns(rewards, dones, gamma):
    returns = []
    discounted_sum = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            discounted_sum = 0
        discounted_sum = reward + gamma * discounted_sum
        returns.insert(0, discounted_sum)
    return np.array(returns)

# Training Loop
for episode in range(max_episodes):
    states, actions, rewards, dones, log_probs = collect_trajectories(env, model)

    old_values = model.critic(torch.tensor(states, dtype=torch.float32)).detach().numpy()
    ppo_update(model, optimizer, states, actions, rewards, dones, log_probs, old_values)
    print(f"Episode {episode + 1}/{max_episodes}, Total Reward: {np.sum(rewards):.2f}")

state = env.reset(goal=np.array([0.1,0.9]))
path = [env.position]

for _ in range(max_steps):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action, _ = model(state_tensor)
    action = action.detach().numpy()
    state, _, done = env.step(action)
    path.append(env.position)
    if done:
        break

path = np.array(path)
plt.plot(path[:, 0], path[:, 1], marker='o')
plt.scatter(env.goal[0], env.goal[1], color='red', label='Goal')
plt.scatter(0.5, 0.1, color='green', label='Start')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Agent's Path")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

state = env.reset(goal=np.array([0.9,0.9]))
path = [env.position]

for _ in range(max_steps):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    action, _ = model(state_tensor)
    action = action.detach().numpy()
    state, _, done = env.step(action)
    path.append(env.position)
    if done:
        break

path = np.array(path)
plt.plot(path[:, 0], path[:, 1], marker='o')
plt.scatter(env.goal[0], env.goal[1], color='red', label='Goal')
plt.scatter(0.5, 0.1, color='green', label='Start')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Agent's Path")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()


def augment_data(self, bc_model, discriminator, dataset):
    self.reset()
    done = False
    disagreement = False
    state_seq = []
    action_seq = []
    padding = np.zeros(6)
    for _ in range(self.seq_len - 1):
        state_seq.append(padding)
        action_seq.append(padding)
    prob = 1
    while not done:
        bc_model.eval()
        discriminator.eval()
        state_seq.append(self.position.copy())
        actions = bc_model(np.array(state_seq).reshape(1, self.seq_len, 6)).detach().cpu().numpy()
        action = actions[0, -1, :]
        action_seq.append(action.copy())
        self.position, done = self.step(action)
        if len(state_seq) == self.seq_len:
            seq = np.concatenate((np.array(state_seq), np.array(action_seq)), axis=1).reshape(1, self.seq_len,
                                                                                              bc_model.state_dim + bc_model.action_dim)
            prob = discriminator(seq)
            prob = prob.item()
            del state_seq[0]
            del action_seq[0]

        if prob <= 0.3:
            new_dataset = self.generate_dataset(initial_position=self.position, n_demo=10, goal_key=self.goal_key)
            dataset = dataset + new_dataset
            print(f"Caught at {self.position} to {self.goals[self.goal_key]}")
            break
    print(f"Reach a goal? steps:{self.steps}, prob:{prob}")
    return dataset
