
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Sampler
import torch
def plot_path(paths, env, save_path):
    fig, axes = plt.subplots(1, len(paths), figsize=(15, 5))  # 1 row, 3 columns
    goal_names = list(env.goals.keys())

    for i, path in enumerate(paths):
        path = np.array(path)  # Convert list of positions to NumPy array
        axes[i].plot(path[:, 0], path[:, 1], marker='.')  # Plot path
        for p in range(len(env.goals.values())):
            axes[i].scatter(env.goals[goal_names[p]][0], env.goals[goal_names[p]][1], color='green')  # Mark goal
        axes[i].scatter(env.goals[goal_names[i]][0], env.goals[goal_names[i]][1], color='red',
                        label="Goal")  # Mark goal
        axes[i].set_title(f"Path to {goal_names[i]}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].legend()
        axes[i].grid(True)

        # Set grid range
        # axes[i].set_xlim(-60, 110)  # X-axis range
        # axes[i].set_ylim(-90, 90)  # Y-axis range

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class SequentialBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, sequence_length):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_samples = len(data_source)

    def __iter__(self):
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = []
            for i in range(start_idx, min(start_idx + self.batch_size, self.num_samples)):
                sequence_indices = list(range(max(0, i - self.sequence_length + 1), i + 1))

                # Ensure sequence length is correct
                while len(sequence_indices) < self.sequence_length:
                    sequence_indices.insert(0, -1)  # Add padding indices

                batch_indices.append(sequence_indices)
            yield batch_indices

    def __len__(self):
        return self.num_samples // self.batch_size
def collate_fn(batch, expert_dataset, state_dim, action_dim):
    """
    Converts batch indices into actual data and handles padding.

    Args:
    - batch: List of index sequences (some indices may be -1 for padding).
    - expert_dataset: Original dataset (Tensor).
    - state_dim: Dimension of state.
    - action_dim: Dimension of action.

    Returns:
    - sequences: Padded tensor of shape (batch_size, sequence_length, feature_dim)
    """
    batch_size = len(batch)
    sequence_length = len(batch[0])  # Sequence length from batch
    feature_dim = state_dim + action_dim

    # Initialize batch with zeros for padding
    padded_batch = torch.zeros((batch_size, sequence_length, feature_dim))

    for i, seq_indices in enumerate(batch):
        for j, idx in enumerate(seq_indices):
            if isinstance(idx, int) and idx != -1:  # Ensure idx is an integer before checking
                padded_batch[i, j] = expert_dataset[idx]  # Copy valid data

    return padded_batch