import numpy as np
import torch
import torch.nn as nn
from alembic.command import current
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
def custom_collate_fn(batch):
    """
    Custom collate function to pad variable-length sequences in a batch.
    Args:
        batch: List of tuples (data, label), where data is a tensor of shape [seq_len, input_dim]
               and label is a scalar tensor.
    Returns:
        Padded batch: (padded_data, labels)
    """
    data, labels = zip(*batch)
    # Pad data to the same length
    padded_data = pad_sequence(data, batch_first=True)  # Shape: [batch_size, max_seq_len, input_dim]
    labels = torch.stack(labels)  # Shape: [batch_size]
    return padded_data, labels

class TrajectoryDataset(Dataset):
    def __init__(self, data, goal_mapping):
        self.data = []
        self.labels = []
        self.goal_mapping = goal_mapping

        for demo in data:
            states = [item[0] for item in demo]
            actions = [item[1] for item in demo]
            goals = [goal_mapping[item[2]] for item in demo]
            states_actions = np.hstack([states, actions])
            self.data.append(torch.tensor(states_actions, dtype=torch.float32))
            self.labels.append(torch.tensor(goals[-1], dtype=torch.long))  # Use the goal of the last step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
# Dataset Generation Function
def generate_dataset(K, noise_range=(-0.5, 0.5), traj_length_range=(30, 80)):
    goals = {
        "g1": np.array([100, 0]),
        "g2": np.array([90, 30]),
        "g3": np.array([40, 60])
    }
    goal_keys = list(goals.keys())
    dataset = []

    for _ in range(K):
        demonstration = []
        current_position = np.array([0, 0])  # Starting position
        remaining_goals = goal_keys[:]
        while remaining_goals:
            current_goal_key = np.random.choice(remaining_goals)
            current_goal = goals[current_goal_key]
            remaining_goals.remove(current_goal_key)

            traj_length = np.random.randint(*traj_length_range)
            base_action = (current_goal - current_position) / traj_length
            for _ in range(traj_length):
                noise = np.random.uniform(*noise_range, size=2)
                action = base_action + noise
                next_position = current_position + action
                demonstration.append((current_position.tolist(), action.tolist(), current_goal_key))
                current_position = next_position
            if not remaining_goals:
                break
        dataset.append(demonstration)
    return dataset

def generate_vanilla_dataset(K, noise_range=(-0.5, 0.5), traj_length_range=(30, 80)):
    goals = {
        "g1": np.array([100, 0]),
        "g2": np.array([90, 30]),
        "g3": np.array([40, 60])
    }
    goal_keys = list(goals.keys())
    dataset = []

    for _ in range(K):
        demonstration = []
        initial_position = np.array([0, 0])  # Starting position
        current_position = np.array([0, 0])  # Starting position
        remaining_goals = goal_keys[:]
        current_goal_key = np.random.choice(remaining_goals)
        current_goal = goals[current_goal_key]
        remaining_goals.remove(current_goal_key)

        traj_length = np.random.randint(*traj_length_range)
        # base_action = (current_goal - current_position) / traj_length
        base_action = (current_goal - initial_position) / traj_length
        for _ in range(traj_length):
            noise = np.random.uniform(*noise_range, size=2)
            action = base_action + noise
            next_position = current_position + action
            demonstration.append((current_position.tolist(), action.tolist(), current_goal_key))
            current_position = next_position
        dataset.append(demonstration)
    return dataset



# Custom Dataset Class
class TrajectoryDataset(Dataset):
    def __init__(self, data, chunk_size, goal_mapping):
        self.data = []
        self.labels = []
        self.chunk_size = chunk_size
        self.goal_mapping = goal_mapping

        for demo in data:
            for i in range(0, len(demo), chunk_size):
                chunk = demo[i:i + chunk_size]
                states = [item[0] for item in chunk]
                actions = [item[1] for item in chunk]
                goals = [goal_mapping[item[2]] for item in chunk]
                chunk_states_actions = np.hstack([states, actions])
                self.data.append(chunk_states_actions)
                self.labels.append(goals[-1])  # Use the goal of the last step

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Transformer Model
class ChunkedTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(ChunkedTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x_last = x[:, -1, :]
        out = self.fc(x_last)
        prob = self.softmax(out)
        return prob

class BCPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BCPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Compute accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def evaluate_bc_policy(bc_policy, test_inputs, test_actions, device):
    """
    Evaluate the trained BC policy on a test dataset.

    Args:
        bc_policy (nn.Module): Trained BC policy model.
        test_inputs (torch.Tensor): Test inputs (state + optional transformer output).
        test_actions (torch.Tensor): Ground truth actions for the test data.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Mean squared error (MSE) on the test dataset.
    """
    bc_policy.eval()
    total_loss = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        # Move inputs and actions to the appropriate device
        test_inputs = test_inputs.to(device)
        test_actions = test_actions.to(device)

        # Forward pass
        predicted_actions = bc_policy(test_inputs)

        # Compute loss
        loss = criterion(predicted_actions, test_actions)
        total_loss += loss.item()

    return total_loss

def prepare_bc_data(data_loader, transformer, device):
    states = []
    transformer_outputs = []
    actions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch  # (inputs: [batch_size, seq_len, input_dim])
            inputs = inputs.to(device)

            # Transformer의 출력 가져오기
            transformer_output = transformer(inputs)  # goal probability
            transformer_outputs.append(transformer_output)

            # state 추출 (마지막 timestep 기준)
            state = inputs[:, -1, :2]  # 현재 state만 추출
            states.append(state)

            # Ground truth action 가져오기
            actions.append(inputs[:, -1, 2:])  # 현재 timestep의 action

    # 데이터 결합
    states = torch.cat(states)
    transformer_outputs = torch.cat(transformer_outputs)
    actions = torch.cat(actions)
    bc_inputs = torch.cat([states, transformer_outputs], dim=1)  # BC 입력 (state + goal probability)
    return bc_inputs, actions

def prepare_bc_g_data(data_loader, transformer, device):
    states = []
    transformer_outputs = []
    actions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch  # (inputs: [batch_size, seq_len, input_dim])
            inputs = inputs.to(device)

            # Transformer의 출력 가져오기
            transformer_output = transformer(inputs)  # goal probability
            _, predicted = torch.max(transformer_output, 1)
            transformer_outputs.append(predicted.reshape(len(predicted), 1))

            # state 추출 (마지막 timestep 기준)
            state = inputs[:, -1, :2]  # 현재 state만 추출
            states.append(state)

            # Ground truth action 가져오기
            actions.append(inputs[:, -1, 2:])  # 현재 timestep의 action

    # 데이터 결합
    states = torch.cat(states)
    transformer_outputs = torch.cat(transformer_outputs)
    actions = torch.cat(actions)
    bc_inputs = torch.cat([states, transformer_outputs], dim=1)  # BC 입력 (state + goal probability)
    return bc_inputs, actions

def prepare_bc_nogoal_data(dataset, device):
    """
    Prepare BC dataset without goal information. It extracts state-action pairs for training BC policy.

    Args:
        dataset (list): A dataset generated by `generate_dataset(K)` containing (state, action, goal) tuples.

    Returns:
        bc_inputs (torch.Tensor): Tensor of states (shape: [N, state_dim]).
        bc_actions (torch.Tensor): Tensor of actions (shape: [N, action_dim]).
    """
    states = []
    actions = []

    # Iterate through demonstrations in the dataset
    for demonstration in dataset:
        for state, action, _ in demonstration:
            states.append(state)  # Extract state
            actions.append(action)  # Extract action

    # Convert to tensors
    bc_inputs = torch.tensor(states, dtype=torch.float32).to(device) # State tensor
    bc_actions = torch.tensor(actions, dtype=torch.float32).to(device)  # Action tensor

    return bc_inputs, bc_actions
import matplotlib.pyplot as plt
import torch

def simulate_trajectory(transformer, bc_policy, initial_state, chunk_size, steps, device):
    transformer.eval()
    bc_policy.eval()

    # Initialize state chunk and trajectory
    state_chunk = [torch.tensor(initial_state + [0, 0], dtype=torch.float32).to(device)]  # Include 2D state + 2D padding
    trajectory = [initial_state]  # For storing (x, y) for visualization

    for _ in range(steps):
        # Create input for ChunkedTransformer
        state_chunk_tensor = torch.stack(state_chunk, dim=0).unsqueeze(0)  # Shape: [1, chunk_len, input_dim]
        state_chunk_tensor = torch.cat(
            [state_chunk_tensor, torch.zeros((1, chunk_size - len(state_chunk), 4), device=device)],
            dim=1,
        )  # Padding if chunk_size > len(state_chunk)

        # Get goal probability from Transformer
        with torch.no_grad():
            goal_probabilities = transformer(state_chunk_tensor)  # Shape: [1, output_dim]

        # Combine current state and Transformer output as BC input
        current_state = state_chunk[-1][:2].unsqueeze(0)  # Extract current state (x, y)
        bc_input = torch.cat([current_state, goal_probabilities], dim=1)  # Shape: [1, 5]

        # Predict next action
        with torch.no_grad():
            action = bc_policy(bc_input).squeeze(0)  # Shape: [2]

        # Update state
        next_state = state_chunk[-1][:2] + action
        trajectory.append(next_state.cpu().tolist())

        # Update state chunk
        if len(state_chunk) >= chunk_size:
            state_chunk.pop(0)
        state_chunk.append(torch.cat([next_state, torch.zeros(2).to(device)]))  # Include padding for next input

    return trajectory
def simulate_g_trajectory(transformer, bc_policy, initial_state, chunk_size, steps, device):
    transformer.eval()
    bc_policy.eval()

    # Initialize state chunk and trajectory
    state_chunk = [torch.tensor(initial_state + [0, 0], dtype=torch.float32).to(device)]  # Include 2D state + 2D padding
    trajectory = [initial_state]  # For storing (x, y) for visualization

    for _ in range(steps):
        # Create input for ChunkedTransformer
        state_chunk_tensor = torch.stack(state_chunk, dim=0).unsqueeze(0)  # Shape: [1, chunk_len, input_dim]
        state_chunk_tensor = torch.cat(
            [state_chunk_tensor, torch.zeros((1, chunk_size - len(state_chunk), 4), device=device)],
            dim=1,
        )  # Padding if chunk_size > len(state_chunk)

        # Get goal probability from Transformer
        with torch.no_grad():
            goal_probabilities = transformer(state_chunk_tensor)  # Shape: [1, output_dim]
            _, predicted = torch.max(goal_probabilities,1)
        # Combine current state and Transformer output as BC input
        current_state = state_chunk[-1][:2].unsqueeze(0)  # Extract current state (x, y)
        bc_input = torch.cat([current_state, predicted.reshape(len(predicted),1)], dim=1)  # Shape: [1, 5]

        # Predict next action
        with torch.no_grad():
            action = bc_policy(bc_input).squeeze(0)  # Shape: [2]

        # Update state
        next_state = state_chunk[-1][:2] + action
        trajectory.append(next_state.cpu().tolist())

        # Update state chunk
        if len(state_chunk) >= chunk_size:
            state_chunk.pop(0)
        state_chunk.append(torch.cat([next_state, torch.zeros(2).to(device)]))  # Include padding for next input

    return trajectory

def simulate_ng_trajectory(bc_policy, initial_state, steps, device):

    bc_policy.eval()

    # Initialize state chunk and trajectory
    trajectory = [initial_state]  # For storing (x, y) for visualization
    current_state = torch.tensor(initial_state, dtype=torch.float32).to(device)
    for _ in range(steps):


        # Predict next action
        with torch.no_grad():
            action = bc_policy(current_state).squeeze(0)  # Shape: [2]

        # Update state
        next_state = current_state + action
        trajectory.append(next_state.cpu().tolist())
        current_state = next_state

    return trajectory

def plot_trajectory(trajectory):
    """
    Plot the trajectory on a 2D plane.

    Args:
        trajectory (list): List of states representing the trajectory [(x1, y1), (x2, y2), ...].
    """
    trajectory = np.array(trajectory)
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], linestyle='-', label='Trajectory')
    plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='Start')  # Start point
    plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='End')  # End point
    goals = {
        "g1": np.array([100, 0]),
        "g2": np.array([90, 30]),
        "g3": np.array([40, 60])
    }
    plt.scatter(goals["g1"][0],goals["g1"][1],color='yellow', label='g1')
    plt.scatter(goals["g2"][0],goals["g2"][1],color='yellow', label='g2')
    plt.scatter(goals["g3"][0],goals["g3"][1],color='yellow', label='g2')
    plt.title('Simulated Trajectory')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    # plt.show()
# Hyperparameters
chunk_size = 5
sequence_length = 50
input_dim = 4  # State (2) + Action (2)
hidden_dim = 128
num_heads = 4
num_layers = 2
output_dim = 3  # Goals: g1, g2, g3
batch_size = 64
learning_rate = 0.001
epochs = 100
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Generate Dataset
K = 1000  # Number of demonstrations
dataset = generate_dataset(K)
# dataset = generate_vanilla_dataset(K)
# Map goal labels to integers
goal_mapping = {"g1": 0, "g2": 1, "g3": 2}

TRAIN_GOAL = False
TRAIN_BC = False
if TRAIN_GOAL:
    # Prepare Dataset and DataLoader
    trajectory_dataset = TrajectoryDataset(dataset, chunk_size, goal_mapping)
    data_loader = DataLoader(trajectory_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Initialize Model, Loss, and Optimizer
    model = ChunkedTransformer(input_dim, hidden_dim, num_heads, num_layers, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


    torch.save(model.state_dict(), "chunked_transformer.pth")
    print("Model saved successfully!")

    if TRAIN_BC:
        bc_policy = BCPolicy(input_dim=5, hidden_dim=128, output_dim=2).to(device)
        optimizer = torch.optim.Adam(bc_policy.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        bc_inputs, bc_actions = prepare_bc_data(data_loader, model, device)

        # BC Policy 학습 루프
        for epoch in range(epochs):
            bc_policy.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = bc_policy(bc_inputs)
            loss = criterion(outputs, bc_actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        torch.save(bc_policy.state_dict(), "bc_goal.pth")
        print("BC Model saved successfully!")
    # Generate a test dataset
    test_K = 10  # Number of test demonstrations
    test_dataset = generate_dataset(test_K)

    # Prepare test DataLoader
    test_trajectory_dataset = TrajectoryDataset(test_dataset, chunk_size, goal_mapping)
    test_data_loader = DataLoader(test_trajectory_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Evaluate the model
    test_accuracy = evaluate_model(model, test_data_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
else:
    loaded_model = ChunkedTransformer(input_dim, hidden_dim, num_heads, num_layers, output_dim)

    # Load the state dictionary
    loaded_model.load_state_dict(torch.load("chunked_transformer.pth"))
    print("Model loaded successfully!")

    # Move to the correct device if needed
    loaded_model = loaded_model.to(device)

    if TRAIN_BC:
        trajectory_dataset = TrajectoryDataset(dataset, chunk_size, goal_mapping)
        data_loader = DataLoader(trajectory_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        criterion = nn.MSELoss()

        bc_policy = BCPolicy(input_dim=5, hidden_dim=128, output_dim=2).to(device)
        optimizer = torch.optim.Adam(bc_policy.parameters(), lr=0.001)
        bc_inputs, bc_actions = prepare_bc_data(data_loader, loaded_model, device)

        bc_g_policy = BCPolicy(input_dim=3, hidden_dim=128, output_dim=2).to(device)
        g_optimizer = torch.optim.Adam(bc_g_policy.parameters(), lr=0.001)
        bc_g_inputs, bc_g_actions = prepare_bc_g_data(data_loader, loaded_model, device)

        bc_ng_policy = BCPolicy(input_dim=2, hidden_dim=128, output_dim=2).to(device)
        ng_optimizer = torch.optim.Adam(bc_ng_policy.parameters(), lr=0.001)
        bc_ng_input, bc_ng_action = prepare_bc_nogoal_data(dataset, device)
        # BC Policy 학습 루프
        epochs = 100
        for epoch in range(epochs):
            # Train BC
            bc_policy.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = bc_policy(bc_inputs)
            loss = criterion(outputs, bc_actions)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Train BC w goal
            bc_g_policy.train()
            g_optimizer.zero_grad()

            # Forward pass
            outputs_g = bc_g_policy(bc_g_inputs)
            loss_g = criterion(outputs_g, bc_g_actions)

            # Backward pass
            loss_g.backward()
            g_optimizer.step()

            ## Train BC no goal
            bc_ng_policy.train()
            ng_optimizer.zero_grad()

            # Forward pass
            outputs_ng = bc_ng_policy(bc_ng_input)
            loss_ng = criterion(outputs_ng, bc_ng_action)

            # Backward pass
            loss_ng.backward()
            ng_optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f} VS  {loss_ng.item():.4f}")
        torch.save(bc_policy.state_dict(), "bc_goal.pth")
        torch.save(bc_g_policy.state_dict(), "bc_g_goal.pth")
        torch.save(bc_ng_policy.state_dict(), "bc_ng_goal.pth")
        print("BC Model saved successfully!")
    # Prepare test DataLoader
    else:
        test_K = 10  # Number of test demonstrations

        test_dataset = generate_dataset(test_K)
        # test_dataset = generate_vanilla_dataset(test_K)

        test_trajectory_dataset = TrajectoryDataset(test_dataset, chunk_size, goal_mapping)
        test_data_loader = DataLoader(test_trajectory_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=custom_collate_fn)
        bc_inputs, bc_actions = prepare_bc_data(test_data_loader, loaded_model, device)
        bc_g_inputs, bc_g_actions = prepare_bc_g_data(test_data_loader, loaded_model, device)
        bc_ng_input, bc_ng_action = prepare_bc_nogoal_data(test_dataset, device)

        loaded_BC_model = BCPolicy(input_dim=5, hidden_dim=128, output_dim=2).to(device)
        loaded_BC_g_model = BCPolicy(input_dim=3, hidden_dim=128, output_dim=2).to(device)
        loaded_BC_ng_model = BCPolicy(input_dim=2, hidden_dim=128, output_dim=2).to(device)

        # Load the state dictionary
        loaded_BC_model.load_state_dict(torch.load("bc_goal.pth"))
        loaded_BC_g_model.load_state_dict(torch.load("bc_g_goal.pth"))
        loaded_BC_ng_model.load_state_dict(torch.load("bc_ng_goal.pth"))
        print("BC Model loaded successfully!")
        loaded_BC_model = loaded_BC_model.to(device)
        loaded_BC_g_model = loaded_BC_g_model.to(device)
        loaded_BC_ng_model = loaded_BC_ng_model.to(device)
    # Evaluate the model
        test_accuracy = evaluate_model(loaded_model, test_data_loader, device)
        print(f"Goal transformer Test Accuracy: {test_accuracy:.2f}%")

        # Evaluate the BC policy
        test_bc_loss = evaluate_bc_policy(loaded_BC_model, bc_inputs, bc_actions, device)
        test_bc_g_loss = evaluate_bc_policy(loaded_BC_g_model, bc_g_inputs, bc_g_actions, device)
        test_bc_ng_loss = evaluate_bc_policy(loaded_BC_ng_model, bc_ng_input, bc_ng_action, device)

        # Print evaluation results
        print(f"Test Loss (MSE): BC :{test_bc_loss:.4f} VS BC wo goal : {test_bc_ng_loss:.4f}")

        initial_state = [0,0]
        chunk_size = 10  # Example chunk size
        steps = 3000  # Total simulation steps

        # Simulate trajectory
        trajectory = simulate_trajectory(loaded_model, loaded_BC_model, initial_state, chunk_size, steps, device)
        trajectory_g = simulate_g_trajectory(loaded_model, loaded_BC_g_model, initial_state, chunk_size, steps, device)

        trajectory_ng = simulate_ng_trajectory(loaded_BC_ng_model, initial_state, steps, device)

        # Visualize trajectory
        plot_trajectory(trajectory)
        plot_trajectory(trajectory_g)

        plot_trajectory(trajectory_ng)
        plt.show()