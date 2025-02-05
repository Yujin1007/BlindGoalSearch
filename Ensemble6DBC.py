import numpy as np
from policies.BC import BCNetwork, TrajectoryDataset,LSTMBC,SequentialTrajectoryDataset
from env.ContinuousGoalEnv import Continuous6DEnsembleEnv
from torch.utils.data import DataLoader
import torch
from env.utils import plot_path
import os
num_samples = 1000
num_ensemble = 5
num_epoch = 10
seq_len=10
state_dim = 6  # Example state dimension
action_dim = 6  # Example action dimension
iter_augmentation = 10
save_path = os.path.join(os.getcwd(), "records")
save_path = os.path.join(save_path, "Ensemble_LSTMBC")
if not os.path.exists(save_path):
    os.makedirs(save_path)
if torch.backends.mps.is_available():
    device="mps"
else:
    device="cpu"

models = []
model = LSTMBC(input_dim=state_dim, output_dim=action_dim, device=device,optim="adam")
    
env = Continuous6DEnsembleEnv(seq_len=seq_len, bc_network=model, goal_reaching_distance=1)
dataset = env.generate_dataset(n_demo=num_samples)

for i in range(num_ensemble):
    model = LSTMBC(input_dim=state_dim, output_dim=action_dim, device=device,optim="adam")
    models.append(model)

for i in range(iter_augmentation):
    expert_dataset = SequentialTrajectoryDataset(dataset, state_dim, action_dim, seq_len=seq_len)
    dataloader = DataLoader(expert_dataset, batch_size=32, shuffle=True)

    # Define the BC model, loss, and optimizer
    for model in models:
        model.train_model(dataloader, num_epochs=num_epoch)
    dataset = env.augment_data(models, dataset)
    paths = env.collect_rollout(models[0])
    save_image_path = os.path.join(save_path, f"{i}.png")
    plot_path(paths, env, save_image_path)

# Test the model
for i, model in enumerate(models):
    paths = env.collect_rollout(model)
    save_image_path = os.path.join(save_path, f"final{i}.png")
    plot_path(paths,env,save_image_path)
for i, model in enumerate(models):
    start_pos = np.array([10.0,0.0])
    paths = env.collect_rollout(model, initial_position=start_pos)
    save_image_path = os.path.join(save_path, f"final{i}_start[10,0].png")
    plot_path(paths,env,save_image_path)
