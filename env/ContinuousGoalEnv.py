import numpy as np
from sklearn.neighbors import KernelDensity
from itertools import combinations


class Continuous2DGridWorld:
    def __init__(self, max_steps=300):
        self.goals = {
            "g1": np.array([100, 0]),
            "g2": np.array([-50, 87]),
            "g3": np.array([-50,-87])
        }
        self.goal_keys = list(self.goals.keys())
        self.reset()
        self.max_steps = max_steps

    def reset(self, initial_position=None, goal_key=None):
        if initial_position is None:
            initial_position = np.array([0.0, 0.0])  # Create a fresh array

        self.position = initial_position
        if goal_key is None:
            self.goal_key = np.random.choice(self.goal_keys)
        else:
            self.goal_key = goal_key
        self.goal =  self.goals[self.goal_key]

        self.steps = 0
        return self.position  # State includes the goal position

    def step(self, action):
        self.position += action
        self.steps += 1
        distance_to_goal = sum(abs(self.position - self.goal))
        # Check if done
        done = self.steps >= self.max_steps or distance_to_goal < 5
        return self.position, done

    def render(self):
        print(f"Agent position: {self.position}, Goal: {self.goal}")

    def generate_dataset(self, initial_position=np.array([0.0,0.0]), n_demo=100, noise_range=(-0.5, 0.5), traj_length_range=(30, 80),goal_key=None):

        dataset = []
        for _ in range(n_demo):
            self.reset(initial_position,goal_key)
            demonstration = []
            current_position = self.position  # Starting position

            traj_length = np.random.randint(*traj_length_range)
            base_action = (self.goal - current_position) / traj_length
            for _ in range(traj_length):
                noise = np.random.uniform(*noise_range, size=2)
                action = base_action + noise
                next_position = current_position + action
                demonstration.append((current_position.tolist(), action.tolist()))
                current_position = next_position

            dataset.append(demonstration)
        return dataset
    def collect_rollout(self, model, initial_position=None):
        paths = []
        model.eval()
        for goal_key in self.goal_keys:
            self.reset(initial_position=initial_position, goal_key=goal_key)
            done = False
            path = [self.position.copy()]
            while not done:
                action = model(self.position)
                self.position, done = self.step(action.detach().numpy())
                path.append(self.position.copy())
            paths.append(path)
        return paths



class Continuous2DEnsembleEnv(Continuous2DGridWorld):
    def augment_data(self, models, dataset):
        self.reset()
        done = False
        path = []
        while not done:
            actions = []
            for model in models:
                model.eval()
                actions.append(model(self.position).detach().numpy())
            self.position, done = self.step(actions[0])
            path.append(self.position)
            disagreement = self.calculate_disagreement(actions)
            if disagreement:
                new_dataset = self.generate_dataset(initial_position=self.position, n_demo=1, goal_key=self.goal_key)
                dataset = dataset + new_dataset
                break
        return dataset

    def calculate_disagreement(self, actions, threshold=2):
        n = len(actions)
        angle_diff_matrix = np.zeros((n, n))  # Pairwise angle difference matrix

        for i in range(n):
            for j in range(i + 1, n):  # Avoid duplicate calculations
                u = actions[i]
                v = actions[j]

                # Calculate cosine similarity
                cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical errors

                # Calculate angle (in rad)
                theta = np.arccos(cos_theta)
                angle_diff_matrix[i, j] = theta
                angle_diff_matrix[j, i] = theta  # Symmetric matrix
        if  np.sum(angle_diff_matrix) > threshold:
            print("disagreement!\n",actions)
            return True
        else:
            return False
class Continuous2DKDEEnv(Continuous2DGridWorld):
    def __init__(self, max_steps=300, bandwidth=1.0):
        super().__init__(max_steps=max_steps)

        self.bandwidth = bandwidth  # KDE의 bandwidth 값
        self.kde_model = KernelDensity(kernel='gaussian', bandwidth=bandwidth)  # KDE 모델 (초기에는 None)

    def setup_kde(self, data):
        self.kde_model.fit(data)
    def augment_data(self, model, dataset, kde_threshold=-10.0):
        self.reset()
        done = False
        while not done:
            model.eval()
            action = model(self.position).detach().numpy()
            self.position, done = self.step(action)
            log_prob = self.kde_model.score_samples(self.position.reshape(1, 2))
            disagreement= log_prob < kde_threshold
            if disagreement:
                new_dataset = self.generate_dataset(initial_position=self.position, n_demo=100, goal_key=self.goal_key)
                dataset = dataset + new_dataset
                print(f"disagreement at {self.position} to {self.goals[self.goal_key]}")
                break
        return dataset

class Continuous6DGridWorld:
    def __init__(self, max_steps=300,goal_reaching_distance=8):
        self.goals = {
            "g1": np.array([43.21, -68.45, 15.12, 82.39, -11.34, 31.11]),
            # "g2": np.array([-76.34, 21.56, -53.89, -25.11, 66.78, -9.88]),
            "g3": np.array([32.23, 45.89, 90.12, -11.12, -45.67, -12.56])
        }
        self.goal_keys = list(self.goals.keys())
        self.trajectories = self.generate_trajectories()
        self.reset()
        self.max_steps = max_steps
        self.goal_reaching_distance = goal_reaching_distance

    def reset(self, initial_position=None, goal_key=None):
        if initial_position is None:
            initial_position = np.zeros(6) # Create a fresh array

        self.position = initial_position
        if goal_key is None:
            self.goal_key = np.random.choice(self.goal_keys)
        else:
            self.goal_key = goal_key
        self.goal =  self.goals[self.goal_key]

        self.steps = 0
        return self.position  # State includes the goal position

    def step(self, action):
        self.position += action
        self.steps += 1
        # distance_to_goal = np.linalg.norm(self.position - self.goal)
        # # Check if done
        # done = self.steps >= self.max_steps or distance_to_goal < self.goal_reaching_distance
        done = self.done(self.position)
        return self.position, done
    def done(self, position):
        distance_to_goal = np.linalg.norm(position - self.goal)
        # Check if done
        done = self.steps >= self.max_steps or distance_to_goal < self.goal_reaching_distance
        return done
    def render(self):
        print(f"Agent position: {self.position}, Goal: {self.goal}")

    def generate_trajectories(self, initial_position=None):
        """
        각 목표(goal)에 대해 주기적인 변화가 포함된 경로를 생성합니다.
        """
        if initial_position is None:
            initial_position = np.zeros(6)
        trajectories = {}
        for goal_key in self.goal_keys:
            trajectory = self.sinusoidal_trajectory(goal_key, initial_position)
            trajectories[goal_key] = np.array(trajectory)  # numpy 배열로 저장
        return trajectories
    def sinusoidal_trajectory(self, goal_key, initial_position=None, n_points=1000):
        if initial_position is None:
            initial_position = np.zeros(6)
        goal = self.goals[goal_key]
        t = np.linspace(0, 1, n_points)  # [0, 1] 구간에서 n_points 생성
        trajectory = []

        for i in range(n_points):
            # 주기적 변화 추가 (사인파 + 랜덤 노이즈)
            freq = np.random.uniform(1, 3, size=6)  # 주파수
            amp = np.random.uniform(1, 3, size=6)  # 진폭

            sinusoidal_effect = amp * np.sin(2 * np.pi * freq * t[i])

            # 선형적으로 목표를 향한 경로 + 주기적 변화
            base_position = (1 - t[i]) * initial_position + t[i] * goal
            if i == n_points-1:
                perturbed_position = base_position
            else:
                perturbed_position = base_position + sinusoidal_effect

            trajectory.append(perturbed_position)
        return trajectory
    # def generate_dataset(self, initial_position=None, n_demo=100, traj_length_range=(30, 35), goal_key=None):
    #
    #     """
    #     미리 생성된 경로(trajectories)에서 점을 샘플링해 데이터를 생성합니다.
    #     """
    #     dataset = []
    #     traj_length = np.random.randint(*traj_length_range)
    #     for _ in range(n_demo):
    #         # 목표를 랜덤 선택
    #         self.reset(initial_position, goal_key)
    #         if initial_position is None:
    #             trajectory = self.trajectories[self.goal_key]
    #         else:
    #             trajectory = self.sinusoidal_trajectory(goal_key=self.goal_key, initial_position=initial_position)
    #             trajectory = np.array(trajectory)
    #         # 경로에서 점 샘플링
    #         # sampled_indices = sorted(np.random.choice(len(trajectory), traj_length, replace=False))
    #         sampled_indices = np.linspace(0, len(trajectory) - 1, traj_length, dtype=int).tolist()
    #         sampled_points = trajectory[sampled_indices]
    #
    #         demonstration = []
    #         # max_action = 0
    #         for i in range(len(sampled_points) - 1):
    #             current_position = sampled_points[i]
    #             next_position = sampled_points[i + 1]
    #             action = next_position - current_position  # next_state = current_state + action
    #             # if max_action < max(abs(action)):
    #             #     max_action = max(abs(action))
    #             #     print(f"max action:{max_action}")
    #             demonstration.append((current_position.flatten().tolist(), action.flatten().tolist()))
    #         demonstration.append((next_position.flatten().tolist(), np.zeros(6).flatten().tolist()))
    #         dataset.append(demonstration)
    #     return dataset
    def generate_dataset(self, initial_position=np.zeros(6), n_demo=100, noise_range=(-0.5, 0.5), traj_length_range=(30, 80),goal_key=None):
        dataset = []
        for _ in range(n_demo):
            self.reset(initial_position,goal_key)
            demonstration = []
            current_position = self.position  # Starting position

            traj_length = np.random.randint(*traj_length_range)
            base_action = (self.goal - current_position) / traj_length
            reached_goal = False
            distance_to_goal = 1
            for _ in range(traj_length):
                noise = np.random.uniform(*noise_range, size=6)
                action = base_action + noise
                next_position = current_position + action
                demonstration.append((current_position.tolist(), action.tolist()))
                current_position = next_position
                distance_to_goal = np.linalg.norm(current_position - self.goal)
                if distance_to_goal < 1:
                    break
            if not reached_goal:
                traj_length = int(distance_to_goal)
                base_action = (self.goal - current_position) / int(distance_to_goal)
                for _ in range(traj_length):
                    next_position = current_position + base_action
                    demonstration.append((current_position.tolist(), base_action.tolist()))
                    current_position = next_position

            dataset.append(demonstration)
        return dataset
    def collect_rollout(self, model, initial_position=None):
        paths = []
        model.eval()
        for goal_key in self.goal_keys:
            self.reset(initial_position=initial_position, goal_key=goal_key)
            done = False
            path = [self.position.copy()]
            while not done:
                action = model(self.position)
                self.position, done = self.step(action.detach().cpu().numpy())
                path.append(self.position.copy())
            paths.append(path)
        return paths
class Continuous6DKDEEnv(Continuous6DGridWorld):
    def __init__(self, max_steps=300, bandwidth=1.0):
        super().__init__(max_steps=max_steps)

        self.bandwidth = bandwidth  # KDE의 bandwidth 값
        self.kde_model = KernelDensity(kernel='gaussian', bandwidth=bandwidth)  # KDE 모델 (초기에는 None)

    def setup_kde(self, data):
        self.kde_model.fit(data)
    def augment_data(self, model, dataset, kde_threshold=-14.0):
        self.reset()
        done = False
        while not done:
            model.eval()
            action = model(self.position).detach().cpu().numpy()
            self.position, done = self.step(action)

            log_prob = self.kde_model.score_samples(self.position.reshape(1, 6))
            disagreement= log_prob < kde_threshold
            if done:
                print(f"done in {self.steps}, log_prob: {log_prob}")
            if disagreement:
                new_dataset = self.generate_dataset(initial_position=self.position, n_demo=100, goal_key=self.goal_key)
                dataset = dataset + new_dataset
                print(f"disagreement at {self.position} to {self.goals[self.goal_key]}")
                break
        return dataset

class Continuous6DGoalAugment(Continuous6DGridWorld):
    def __init__(self, bc_network, seq_len, max_steps=300):
        super().__init__(max_steps=max_steps)
        self.seq_len = seq_len
        self.bc_lstm = False
        for module in bc_network.modules():
            if isinstance(module, nn.LSTM):
                self.bc_lstm = True
    def augment_data(self, model, dataset):
        self.reset()
        done = False
        disagreement = False
        state_seq = []
        action_seq = []
        padding = np.zeros(6)
        for _ in range(self.seq_len - 1):
            state_seq.append(padding)
            action_seq.append(padding)

        while not done:
            model.eval()
            if self.bc_lstm:
                state_seq.append(self.position.copy())
                actions = model(np.array(state_seq).reshape(1, self.seq_len, 6)).detach().cpu().numpy()
                action = actions[0, -1, :]
                action_seq.append(action.copy())
                if len(state_seq) == self.seq_len:
                    del state_seq[0]
            else:
                action = model(self.position).detach().cpu().numpy()
            self.position, done = self.step(action)
            if not done:
                for any_goal in self.goals.values():
                    distance_to_goal = np.linalg.norm(self.position - any_goal)
                    # Check if done
                    disagreement = distance_to_goal < self.goal_reaching_distance
                    break

            if disagreement:
                new_dataset = self.generate_dataset(initial_position=self.position, n_demo=100, goal_key=self.goal_key)
                dataset = dataset + new_dataset
                print(f"disagreement at {self.position} to {self.goals[self.goal_key]}")
                break
        print("Reach a goal? ",disagreement)
        return dataset
    def generate_dataset(self, initial_position=np.zeros(6), n_demo=100, noise_range=(-0.5, 0.5),
                         traj_length_range=(30, 80), goal_key=None):
        dataset = []
        for _ in range(n_demo):
            self.reset(initial_position, goal_key)
            demonstration = []
            current_position = self.position  # Starting position
            padding = np.zeros(6).tolist()
            for _ in range(self.seq_len-1):
                demonstration.append((padding, padding))
            traj_length = np.random.randint(*traj_length_range)
            base_action = (self.goal - current_position) / traj_length
            reached_goal = False
            distance_to_goal = 1
            for _ in range(traj_length):
                noise = np.random.uniform(*noise_range, size=6)
                action = base_action + noise
                next_position = current_position + action
                demonstration.append((current_position.tolist(), action.tolist()))
                current_position = next_position
                distance_to_goal = np.linalg.norm(current_position - self.goal)
                if distance_to_goal < 1:
                    break
            if not reached_goal:
                traj_length = int(distance_to_goal)
                base_action = (self.goal - current_position) / int(distance_to_goal)
                for _ in range(traj_length):
                    next_position = current_position + base_action
                    demonstration.append((current_position.tolist(), base_action.tolist()))
                    current_position = next_position

            dataset.append(demonstration)
        return dataset
    def collect_rollout(self, model, initial_position=None):
        paths = []
        model.eval()

        for goal_key in self.goal_keys:
            state_seq = []
            padding = np.zeros(6)
            for _ in range(self.seq_len - 1):
                state_seq.append(padding)
            self.reset(initial_position=initial_position, goal_key=goal_key)
            done = False
            path = [self.position.copy()]
            while not done:
                state_seq.append(self.position.copy())
                actions = model(np.array(state_seq).reshape(1, self.seq_len, 6)).detach().cpu().numpy()
                action = actions[0, -1, :]
                self.position, done = self.step(action)
                if len(state_seq) == self.seq_len:
                    del state_seq[0]
                path.append(self.position.copy())

            paths.append(path)
        return paths


class Continuous6DEnsembleEnv(Continuous6DGridWorld):
    def __init__(self, bc_network, max_steps=300, seq_len=None,goal_reaching_distance=8):
        super().__init__(max_steps=max_steps)
        self.seq_len = seq_len
        self.bc_lstm = False
        for module in bc_network.modules():
            if isinstance(module, nn.LSTM):
                self.bc_lstm = True
    
    def augment_data(self, models, dataset):
        self.reset()
        done = False
        path = []
        while not done:
            actions = []
            for model in models:
                model.eval()
                actions.append(model(self.position).detach().numpy())
            self.position, done = self.step(actions[0])
            path.append(self.position)
            disagreement = self.calculate_disagreement(actions)
            if disagreement:
                new_dataset = self.generate_dataset(initial_position=self.position, n_demo=1, goal_key=self.goal_key)
                dataset = dataset + new_dataset
                break
        return dataset
    def augment_data(self, models, dataset):
        self.reset()
        done = False
        disagreement = False
        state_seq = []
        padding = np.zeros(6)
        for _ in range(self.seq_len - 1):
            state_seq.append(padding)

        while not done:
            actions = []
            for model in models:
                model.eval()
                if self.bc_lstm:
                    state_seq.append(self.position.copy())
                    action_seq = model(np.array(state_seq).reshape(1, self.seq_len, 6)).detach().cpu().numpy()
                    action = action_seq[0, -1, :]
                    if len(state_seq) == self.seq_len:
                        del state_seq[0]

                else:
                    action = model(self.position).detach().cpu().numpy()
                actions.append(action)
                
            self.position, done = self.step(actions[0])
            disagreement = self.calculate_disagreement(actions)
            
            if disagreement:
                new_dataset = self.generate_dataset(initial_position=self.position, n_demo=100, goal_key=self.goal_key)
                dataset = dataset + new_dataset
                print(f"disagreement at {self.position} to {self.goals[self.goal_key]}")
                break
        print("Reach a goal? ",disagreement)
        return dataset
    
    def calculate_disagreement(self, actions, threshold=0.5):
        angle_differences = []
        for v1, v2 in combinations(actions, 2):
            cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cosine_similarity, -1.0, 1.0))  # arccos의 안정성을 위해 clip 사용
            angle_differences.append(angle)

        # 평균 각도 차이
        mean_angle_difference = np.mean(angle_differences)
        if  mean_angle_difference > threshold:
            print("disagreement!\n",actions)
            return True
        else:
            return False
    def generate_dataset(self, initial_position=np.zeros(6), n_demo=100, noise_range=(-0.5, 0.5),
                         traj_length_range=(30, 80), goal_key=None):
        dataset = []
        for _ in range(n_demo):
            self.reset(initial_position, goal_key)
            demonstration = []
            current_position = self.position  # Starting position
            padding = np.zeros(6).tolist()
            for _ in range(self.seq_len-1):
                demonstration.append((padding, padding))
            traj_length = np.random.randint(*traj_length_range)
            base_action = (self.goal - current_position) / traj_length
            reached_goal = False
            distance_to_goal = 1
            for _ in range(traj_length):
                noise = np.random.uniform(*noise_range, size=6)
                action = base_action + noise
                next_position = current_position + action
                demonstration.append((current_position.tolist(), action.tolist()))
                current_position = next_position
                distance_to_goal = np.linalg.norm(current_position - self.goal)
                if distance_to_goal < 1:
                    break
            if not reached_goal:
                traj_length = int(distance_to_goal)
                base_action = (self.goal - current_position) / int(distance_to_goal)
                for _ in range(traj_length):
                    next_position = current_position + base_action
                    demonstration.append((current_position.tolist(), base_action.tolist()))
                    current_position = next_position

            dataset.append(demonstration)
        return dataset
    def collect_rollout(self, model, initial_position=None):
        paths = []
        model.eval()

        for goal_key in self.goal_keys:
            state_seq = []
            padding = np.zeros(6)
            for _ in range(self.seq_len - 1):
                state_seq.append(padding)
            self.reset(initial_position=initial_position, goal_key=goal_key)
            done = False
            path = [self.position.copy()]
            while not done:
                state_seq.append(self.position.copy())
                actions = model(np.array(state_seq).reshape(1, self.seq_len, 6)).detach().cpu().numpy()
                action = actions[0, -1, :]
                self.position, done = self.step(action)
                if len(state_seq) == self.seq_len:
                    del state_seq[0]
                path.append(self.position.copy())

            paths.append(path)
        return paths

import torch.nn as nn
class Continuous6DGAN(Continuous6DGridWorld):
    def __init__(self, bc_network, max_steps=300, seq_len=None,goal_reaching_distance=8):
        super().__init__(max_steps=max_steps)
        self.seq_len = seq_len
        self.bc_lstm = False
        for module in bc_network.modules():
            if isinstance(module, nn.LSTM):
                self.bc_lstm = True
    def augment_data(self, bc_model, discriminator, dataset):
        self.reset()
        done = False
        disagreement = False
        state_seq = []
        action_seq = []
        padding = np.zeros(6)
        for _ in range(self.seq_len-1):
            state_seq.append(padding)
            action_seq.append(padding)
        prob = 1
        while not done:
            bc_model.eval()
            discriminator.eval()
            state_seq.append(self.position.copy())
            actions = bc_model(np.array(state_seq).reshape(1,self.seq_len, 6)).detach().cpu().numpy()
            action = actions[0,-1,:]
            action_seq.append(action.copy())
            self.position, done = self.step(action)
            if len(state_seq) == self.seq_len:
                seq = np.concatenate((np.array(state_seq), np.array(action_seq)), axis=1).reshape(1, self.seq_len, bc_model.state_dim + bc_model.action_dim)
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

    def generate_dataset(self, initial_position=np.zeros(6), n_demo=100, noise_range=(-0.5, 0.5),
                         traj_length_range=(30, 80), goal_key=None):
        dataset = []
        for _ in range(n_demo):
            self.reset(initial_position, goal_key)
            demonstration = []
            current_position = self.position  # Starting position
            padding = np.zeros(6).tolist()
            for _ in range(self.seq_len-1):
                demonstration.append((padding, padding))
            traj_length = np.random.randint(*traj_length_range)
            base_action = (self.goal - current_position) / traj_length
            reached_goal = False
            distance_to_goal = 1
            for _ in range(traj_length):
                noise = np.random.uniform(*noise_range, size=6)
                action = base_action + noise
                next_position = current_position + action
                demonstration.append((current_position.tolist(), action.tolist()))
                current_position = next_position
                distance_to_goal = np.linalg.norm(current_position - self.goal)
                if distance_to_goal < 1:
                    break
            if not reached_goal:
                traj_length = int(distance_to_goal)
                base_action = (self.goal - current_position) / int(distance_to_goal)
                for _ in range(traj_length):
                    next_position = current_position + base_action
                    demonstration.append((current_position.tolist(), base_action.tolist()))
                    current_position = next_position

            dataset.append(demonstration)
        return dataset
    def collect_rollout(self, model, initial_position=None):
        paths = []
        model.eval()

        for goal_key in self.goal_keys:
            state_seq = []
            padding = np.zeros(6)
            for _ in range(self.seq_len - 1):
                state_seq.append(padding)
            self.reset(initial_position=initial_position, goal_key=goal_key)
            done = False
            path = [self.position.copy()]
            while not done:
                state_seq.append(self.position.copy())
                actions = model(np.array(state_seq).reshape(1, self.seq_len, 6)).detach().cpu().numpy()
                action = actions[0, -1, :]
                self.position, done = self.step(action)
                if len(state_seq) == self.seq_len:
                    del state_seq[0]
                path.append(self.position.copy())

            paths.append(path)
        return paths