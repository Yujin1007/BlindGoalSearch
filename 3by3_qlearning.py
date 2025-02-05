import numpy as np
import random
import matplotlib.pyplot as plt

# Environment parameters
grid_size = (3, 3)
start_position = (1, 0)
goal_positions = [(2, 2),(0,2)]
actions = ['up', 'down', 'left', 'right']
action_mapping = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}
max_steps = 10
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon-greedy policy
episodes = 10000

# Initialize Q-table
Q_table = np.zeros((*grid_size, len(actions)))


# Helper functions
def is_valid_position(position):
    return 0 <= position[0] < grid_size[0] and 0 <= position[1] < grid_size[1]


def get_next_position(current_position, action):
    delta = action_mapping[action]
    next_position = (current_position[0] + delta[0], current_position[1] + delta[1])
    if is_valid_position(next_position):
        return next_position
    return current_position  # If invalid move, stay in the same position


def get_reward(goal, position, step_count):
    if position == goal:
        return 1 - 0.9 * (step_count / max_steps)
    return 0



# Visualizing the Q-table
def visualize_q_table(Q_table):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, grid_size[1])
    ax.set_ylim(0, grid_size[0])
    ax.set_xticks(np.arange(grid_size[1] + 1))
    ax.set_yticks(np.arange(grid_size[0] + 1))
    ax.grid(True)
    ax.set_aspect('equal')

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            q_values = Q_table[i, j]

            # Coordinates of the cell corners
            x_left, x_right = j, j + 1
            y_top, y_bottom = grid_size[0] - i, grid_size[0] - i - 1

            # Define the triangles for each action
            triangles = {
                'up': [(x_left, y_top), (x_right, y_top), ((x_left + x_right) / 2, (y_top + y_bottom) / 2)],
                'right': [(x_right, y_top), (x_right, y_bottom), ((x_left + x_right) / 2, (y_top + y_bottom) / 2)],
                'down': [(x_left, y_bottom), (x_right, y_bottom), ((x_left + x_right) / 2, (y_top + y_bottom) / 2)],
                'left': [(x_left, y_top), (x_left, y_bottom), ((x_left + x_right) / 2, (y_top + y_bottom) / 2)],
            }

            # Place Q-values at the center of each triangle
            centers = {
                'up': ((x_left + x_right) / 2, y_top - 0.25),
                'right': (x_right - 0.25, (y_top + y_bottom) / 2),
                'down': ((x_left + x_right) / 2, y_bottom + 0.25),
                'left': (x_left + 0.25, (y_top + y_bottom) / 2),
            }

            for action_idx, action in enumerate(actions):
                # Draw the triangle
                triangle = triangles[action]
                ax.fill(*zip(*triangle), color='lightblue', edgecolor='black', alpha=0.3)

                # Add the Q-value text
                center = centers[action]
                ax.text(center[0], center[1], f"{q_values[action_idx]:.2f}",
                        ha='center', va='center', fontsize=8, color='black')

    plt.title("Q-table Visualization")
    plt.gca().invert_yaxis()
    plt.show()

# Simulating a path using the learned policy
def simulate_path(start, goal):
    position = start
    path = [position]

    for _ in range(max_steps):
        action_idx = np.argmax(Q_table[position[0], position[1]])
        q_max = np.max(Q_table[position[0], position[1]])
        action = actions[action_idx]
        next_position = get_next_position(position, action)
        path.append((next_position,np.round(q_max,2)))

        if next_position == goal:
            break
        position = next_position

    return path

goal_22 = 0
goal_02 = 0

# Q-learning algorithm
for episode in range(episodes):
    position = start_position
    goal = random.choice(goal_positions)
    step_count = 0
    if goal == (2,2):
        goal_22 += 1
    else:
        goal_02 += 1

    for _ in range(max_steps):
        step_count += 1

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, len(actions) - 1)
        else:
            action_idx = np.argmax(Q_table[position[0], position[1]])

        action = actions[action_idx]
        next_position = get_next_position(position, action)
        reward = get_reward(goal, next_position, step_count)

        # Q-value update
        best_next_q = np.max(Q_table[next_position[0], next_position[1]])
        Q_table[position[0], position[1], action_idx] += alpha * (
                reward + gamma * best_next_q - Q_table[position[0], position[1], action_idx]
        )

        position = next_position

        if position == goal:
            break
print(f"goal 02 : {goal_02}, || goal 22 : {goal_22}")
path = simulate_path(start_position, (0,2))
print(f"goal:(0,2))")
print(f"path: {path}")
path = simulate_path(start_position, (2,2))
print(f"goal:(2,2))")
print(f"path: {path}")
visualize_q_table(Q_table)
# Visualizing the agent's path
# goal = random.choice(goal_positions)

# grid = np.zeros(grid_size)
# for pos in path:
#     grid[pos] = 0.5  # Mark the path
# grid[goal] = 1  # Mark the goal
#
# plt.figure(figsize=(6, 6))
# plt.imshow(grid, cmap='Blues', origin='upper')
# plt.colorbar(label='Path Intensity')
# plt.title("Agent's Path")
# plt.xticks(range(grid_size[1]))
# plt.yticks(range(grid_size[0]))
# plt.scatter(start_position[1], start_position[0], color='green', label='Start')
# plt.scatter(goal[1], goal[0], color='red', label='Goal')
# plt.legend()
# plt.show()