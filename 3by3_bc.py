import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Environment parameters
grid_size = (3, 3)
actions = ['up', 'down', 'left', 'right']
action_mapping = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
inverse_action_mapping = {v: k for k, v in action_mapping.items()}

# Demonstrations
# demonstrations = [
#     [(1, 0), (1, 1), (1, 2), (0, 2)],  # Demo 1
#     [(1, 0), (1, 1), (1, 2), (2, 2)],
#     [(1, 0), (1, 1), (1, 2), (2, 2)]# Demo 2
# ]
# demonstrations = [
#     [(1, 0), (0, 0), (0, 1), (0, 2)],  # Demo 1
#     [(1, 0), (2, 0), (2, 1), (2, 2)]  # Demo 2
# ]
# Create dataset from demonstrations
X = []
y = []


def generate_demonstrations(grid_size, goal):
    demonstrations = []
    for start_row in range(grid_size[0]):
        for start_col in range(grid_size[1]):
            path = []
            current_pos = (start_row, start_col)

            # Move vertically towards the goal
            while current_pos[0] != goal[0]:
                next_pos = (current_pos[0] + (1 if goal[0] > current_pos[0] else -1), current_pos[1])
                path.append(current_pos)
                current_pos = next_pos

            # Move horizontally towards the goal
            while current_pos[1] != goal[1]:
                next_pos = (current_pos[0], current_pos[1] + (1 if goal[1] > current_pos[1] else -1))
                path.append(current_pos)
                current_pos = next_pos

            # Add the goal to the path
            path.append(goal)
            demonstrations.append(path)
    return demonstrations
def is_valid_position(position):
    return 0 <= position[0] < grid_size[0] and 0 <= position[1] < grid_size[1]


def get_next_position(current_position, action):
    delta = action_mapping[action]
    next_position = (current_position[0] + delta[0], current_position[1] + delta[1])
    if is_valid_position(next_position):
        return next_position
    return current_position  # If invalid move, stay in the same position

goal_position = (2, 2)
demonstration1 = generate_demonstrations(grid_size, goal_position)
goal_position = (0, 2)
demonstration2 = generate_demonstrations(grid_size, goal_position)
demonstrations = demonstration1 + demonstration2
print(demonstrations)
random.shuffle(demonstrations)
# print(demonstrations)
for demo in demonstrations:
    for i in range(len(demo) - 1):
        state = demo[i]
        next_state = demo[i + 1]
        action = None
        for a, move in action_mapping.items():
            if (state[0] + (1 if a == 'down' else -1 if a == 'up' else 0),
                state[1] + (1 if a == 'right' else -1 if a == 'left' else 0)) == next_state:
                action = move
        X.append(state)
        y.append(action)

X = np.array(X) / np.array(grid_size)  # Normalize state inputs
y = np.array(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)


# Define a simple neural network
class BCNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(BCNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.fc(x)


# Instantiate and train the model
input_size = 2  # (x, y) coordinates
output_size = len(actions)
model = BCNet(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()


# Simulate a path using the learned model
def simulate_path_bc(start_position, goal):
    position = start_position
    path = [position]

    for _ in range(10):  # Max steps
        state = torch.tensor([np.array(position) / np.array(grid_size)], dtype=torch.float32)
        with torch.no_grad():
            print(f"state {state}, model out {model(state)}")
            action_idx = model(state).argmax(dim=1).item()
        action = inverse_action_mapping[action_idx]

        # Calculate next position
        delta = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }[action]
        next_position = (position[0] + delta[0], position[1] + delta[1])
        # next_position = get_next_position(position, action)
        if not is_valid_position(next_position):
            next_position = position
        # if next_position not in path:
        path.append(next_position)
        if next_position == goal:
            break
        position = next_position

    return path


# Visualize the path
start_position = (1, 0)
path = simulate_path_bc(start_position, (0,2))
print("goal (0,2) :", path)
path = simulate_path_bc(start_position, (2,2))
print("goal (2,2) :", path)
grid = np.zeros(grid_size)
for pos in path:
    grid[pos] = 0.5  # Mark the path
grid[start_position] = 1  # Start
grid[path[-1]] = 0.8  # End

plt.figure(figsize=(6, 6))
plt.imshow(grid, cmap='Blues', origin='upper')
plt.title("Simulated Path by BC Agent")
plt.xticks(range(grid_size[1]))
plt.yticks(range(grid_size[0]))
plt.scatter(start_position[1], start_position[0], color='green', label='Start')
plt.scatter(path[-1][1], path[-1][0], color='red', label='Goal')
plt.legend()
plt.show()