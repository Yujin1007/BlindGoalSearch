import copy
import numpy as np

from policies.BC import BCNetwork,BCDiscriminator, LSTMBC, SequentialTrajectoryDataset
from env.ContinuousGoalEnv import Continuous6DGAN
from torch.utils.data import DataLoader
from env.utils import SequentialBatchSampler, collate_fn
import torch
from env.utils import plot_path
import os
from policies.GAN import LSTMDiscriminator
import matplotlib.pyplot as plt
num_samples = 1000
seq_len=10
num_epoch = 10 #20
state_dim = 6
action_dim = 6
iter_augmentation = 10
total_iteration = 1000
batch_size=32

save_path = os.path.join(os.getcwd(), "records")
save_path = os.path.join(save_path, "GAN_LSTMBC_weakD")
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

# bc = BCNetwork(input_dim=state_dim, output_dim=action_dim, device=device,optim="adam")
bc = LSTMBC(input_dim=state_dim, output_dim=action_dim, device=device,optim="adam")
discriminator=LSTMDiscriminator(state_dim=state_dim,action_dim=action_dim,sequence_length=seq_len, device=device)
env = Continuous6DGAN(seq_len=seq_len, bc_network=bc, goal_reaching_distance=1)

model = BCDiscriminator(bc, discriminator, env, seq_len, device=device)

# bc.load_state_dict(torch.load(f"{save_path}/BC.pth"))
# paths = env.collect_rollout(model.bc_network)
# save_image_path = os.path.join(save_path, f"final.png")
# plot_path(paths, env, save_image_path)


dataset = env.generate_dataset(n_demo=num_samples)
expert_dataset = SequentialTrajectoryDataset(dataset, state_dim, action_dim, seq_len)
dataloader = DataLoader(expert_dataset, batch_size=32, shuffle=True)
# dataloader = DataLoader(
#     expert_dataset,
#     batch_sampler=SequentialBatchSampler(expert_dataset, batch_size, sequence_length),
#     collate_fn=lambda batch: collate_fn(batch, expert_dataset, state_dim, action_dim),
#     shuffle=False
# )
model.train_model(dataloader, num_epochs=num_epoch)

#Train with augmented data
# for i in range(iter_augmentation):
i=0
i_total = 0
while i != iter_augmentation:
    len_original = len(copy.copy(dataset))
    dataset = env.augment_data(model.bc_network, model.discriminator, dataset)
    len_augmented = len(copy.copy(dataset))
    if len_original != len_augmented:
        i+= 1
    expert_dataset = SequentialTrajectoryDataset(dataset, state_dim, action_dim, seq_len)
    dataloader = DataLoader(expert_dataset, batch_size=32, shuffle=True)

    model.train_model(dataloader, num_epochs=num_epoch)

    paths = env.collect_rollout(model.bc_network)
    save_image_path = os.path.join(save_path, f"{i}.png")
    plot_path(paths, env, save_image_path)
    check_done = []
    for p in paths:
        if len(p) >= env.max_steps:
            check_done.append(False)
        else:
            check_done.append(True)
    if np.array(check_done).all():
        break

    i_total += 1
    if i_total == total_iteration:
        break
# Test the model
model_path = os.path.join(save_path, "BC.pth")
torch.save(model.bc_network.state_dict(), model_path)
model_path = os.path.join(save_path, "Discriminator.pth")
torch.save(model.discriminator.state_dict(), model_path)

paths = env.collect_rollout(model.bc_network)
save_image_path = os.path.join(save_path, f"final.png")
plot_path(paths, env, save_image_path)