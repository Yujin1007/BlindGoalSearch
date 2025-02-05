import copy

import numpy as np
from policies.BC import BCNetwork,LSTMBC, TrajectoryDataset,SequentialTrajectoryDataset
from env.ContinuousGoalEnv import Continuous6DGoalAugment
from torch.utils.data import DataLoader
import torch
from env.utils import plot_path
import os
import matplotlib.pyplot as plt
num_samples = 1000
num_ensemble = 5
num_epoch = 10
state_dim = 6  # Example state dimension
action_dim = 6  # Example action dimension
iter_augmentation = 10
iter_total = 1000
seq_len = 10
save_path = os.path.join(os.getcwd(), "records")
save_path = os.path.join(save_path, "anygoal_LSTM_BC")
if not os.path.exists(save_path):
    os.makedirs(save_path)
if torch.backends.mps.is_available():
    device="mps"
else:
    device="cpu"
# device="cpu"


# middle_data1=env.generate_dataset(n_demo=30, initial_position = np.array([100,0]), goal_key="g2")
# middle_data2=env.generate_dataset(n_demo=1, initial_position = np.array([100,0]), goal_key="g3")
# dataset = dataset + middle_data1 + middle_data2
#
# expert_dataset = TrajectoryDataset(dataset)
# states = expert_dataset.states
# kde = KernelDensity(kernel='gaussian', bandwidth=1.0)  # bandwidth는 적절히 조정
# kde.fit(states)
# new_states = np.array([[110.0,.0],[-55.0, 93.0],[-55., -93.]] )
# exiting_states = np.array([[100.,0.],[-50.,87],[-50.,87.]])
#
# point1 = env.goals["g1"]
# point2 = env.goals["g2"]
# point3 = env.goals["g3"]
#
# # 선분 위의 점 3개 추출
# num_points = 3
# points_on_line_g12 = np.linspace(point1, point2, num_points + 2)[1:-1]  # 끝점 제외하고 중간 3개 추출
# points_on_line_g13 = np.linspace(point1, point3, num_points + 2)[1:-1]  # 끝점 제외하고 중간 3개 추출
#
#
# for ns in new_states:
#     log_prob = kde.score_samples(ns.reshape(1,2))
#     print(f"Log Probability for new states: {log_prob}")
# print("//////")
# for es in points_on_line_g12:
#     log_prob = kde.score_samples(es.reshape(1,2))
#     print(f"{es}: {log_prob}")
# print("//////")
# for es in points_on_line_g13:
#     log_prob = kde.score_samples(es.reshape(1, 2))
#     print(f"{es}: {log_prob}")
# print("//////")
# increasing_states = np.array([[x, 0] for x in range(90, 111)])
# for ins in increasing_states:
#     log_prob = kde.score_samples(ins.reshape(1, 2))
#     print(f"{ins}: {log_prob}")

# model = BCNetwork(input_dim=state_dim, output_dim=action_dim, device=device)
model = LSTMBC(input_dim=state_dim, output_dim=action_dim, device=device,optim="adam")
env = Continuous6DGoalAugment(seq_len=seq_len, bc_network=model)

dataset = env.generate_dataset(n_demo=num_samples)
# expert_dataset = TrajectoryDataset(dataset)
# dataloader = DataLoader(expert_dataset, batch_size=32, shuffle=False)
expert_dataset = SequentialTrajectoryDataset(dataset, state_dim, action_dim, seq_len)
dataloader = DataLoader(expert_dataset, batch_size=32, shuffle=True)


# env.setup_kde(expert_dataset.states)
#initialize BC

# actions = expert_dataset.actions
# # Flatten the tensor to find the maximum in the entire tensor
# max_value, max_index = actions.view(-1).max(dim=0)
#
# # Convert the flat index to 2D coordinates
# position = torch.unravel_index(max_index, actions.shape)
# print(f"Maximum Value: {max_value}")
# print(f"Position: {position}")
# for i in range(10):
#     X = []
#     Y = []
#     for x in range(len(dataset[i])):
#         X.append(dataset[position[0].item()][x][0][position[1].item()])
#         Y.append(dataset[i][x][0][1])
#     plt.scatter(np.linspace(0,len(X)-1,len(X)),X)
#     plt.plot(np.linspace(0,len(X)-1,len(X)),X)
#     plt.show()
model.train_model(dataloader, num_epochs=num_epoch)

#Train with augmented data
# for i in range(iter_augmentation):
i=0
i_total = 0
while i != iter_augmentation:
    len_original = len(copy.copy(dataset))
    dataset = env.augment_data(model, dataset)
    len_augmented = len(copy.copy(dataset))
    if len_original != len_augmented:
        i+= 1
    # expert_dataset = TrajectoryDataset(dataset)
    # dataloader = DataLoader(expert_dataset, batch_size=128, shuffle=False)
    expert_dataset = SequentialTrajectoryDataset(dataset, state_dim, action_dim, seq_len)
    dataloader = DataLoader(expert_dataset, batch_size=32, shuffle=True)

    model.train_model(dataloader, num_epochs=num_epoch)

    paths = env.collect_rollout(model)
    save_image_path = os.path.join(save_path, f"{i}.png")
    plot_path(paths, env, save_image_path)
    check_done = []
    i_total += 1
    if i_total == iter_total:
        print("iteration over")
        break
    for p in paths:
        if len(p) >= env.max_steps:
            check_done.append(False)
        else:
            check_done.append(True)
    if np.array(check_done).all():
        break

model_path = os.path.join(save_path, "BC.pth")
torch.save(model.bc_network.state_dict(), model_path)

paths = env.collect_rollout(model.bc_network)
save_image_path = os.path.join(save_path, f"final.png")
plot_path(paths, env, save_image_path)