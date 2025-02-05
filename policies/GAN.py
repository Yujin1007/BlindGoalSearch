import torch
import torch.nn as nn
import torch.optim as optim

class GANInverseRL(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3):
        super(GANInverseRL, self).__init__()

        # 🔹 Define Generator (Policy Network)
        self.generator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output action in range (-1,1)
        )

        # 🔹 Define Discriminator (Expert vs. Agent Classifier)
        self.odiscriminatr = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of being an expert action
        )

        # 🔹 Define Optimizers
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)

        # 🔹 Loss Function (Binary Cross Entropy for Discriminator)
        self.criterion = nn.BCELoss()

    def generate_action(self, state):
        """ Forward pass through the Generator to get an action. """
        return self.generator(state)

    def discriminate(self, state, action):
        """ Forward pass through the Discriminator to classify (s,a) pairs. """
        x = torch.cat((state, action), dim=-1)  # Concatenate state & action
        return self.discriminator(x)

    def train_step(self, expert_states, expert_actions, batch_size=32):
        """ Performs one training step for both Generator and Discriminator. """

        # ============================
        # 🔹 1. Train the Discriminator
        # ============================
        self.disc_optimizer.zero_grad()

        # Labels: 1 for expert data, 0 for agent-generated data
        expert_labels = torch.ones((batch_size, 1))  # Expert → 1
        agent_labels = torch.zeros((batch_size, 1))  # Agent → 0

        # Generate agent actions using the Generator
        agent_states = expert_states  # Assume same distribution for fair comparison
        agent_actions = self.generate_action(agent_states)

        # Forward pass through the Discriminator
        expert_preds = self.discriminate(expert_states, expert_actions)  # Expert (should be 1)
        agent_preds = self.discriminate(agent_states, agent_actions)  # Agent (should be 0)

        # Compute Discriminator Loss
        loss_expert = self.criterion(expert_preds, expert_labels)
        loss_agent = self.criterion(agent_preds, agent_labels)
        disc_loss = loss_expert + loss_agent

        # Backpropagation for Discriminator
        disc_loss.backward()
        self.disc_optimizer.step()

        # ============================
        # 🔹 2. Train the Generator
        # ============================
        self.gen_optimizer.zero_grad()

        # Generate new actions from Generator
        gen_actions = self.generate_action(agent_states)

        # Compute Generator loss (tries to fool the Discriminator)
        gen_preds = self.discriminate(agent_states, gen_actions)
        gen_loss = self.criterion(gen_preds, expert_labels)  # Wants discriminator to classify it as expert

        # Backpropagation for Generator
        gen_loss.backward()
        self.gen_optimizer.step()

        return disc_loss.item(), gen_loss.item()

    def compute_reward(self, state, action):
        """ Computes the reward from the Discriminator as -log(D(s, a)). """
        with torch.no_grad():
            reward = -torch.log(self.discriminate(state, action) + 1e-8)  # Prevent log(0)
        return reward

    def train_gan(self, expert_states, expert_actions, epochs=100, batch_size=32):
        """ Train the GAN model with expert demonstrations. """

        num_samples = expert_states.shape[0]
        for epoch in range(epochs):
            # Sample a random batch
            indices = torch.randint(0, num_samples, (batch_size,))
            batch_expert_states = expert_states[indices]
            batch_expert_actions = expert_actions[indices]

            # Train one step
            disc_loss, gen_loss = self.train_step(batch_expert_states, batch_expert_actions, batch_size)

            if epoch % 10 == 0:  # Print loss every 10 epochs
                print(f"Epoch {epoch}/{epochs}, Disc Loss: {disc_loss:.4f}, Gen Loss: {gen_loss:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3):
        super(Discriminator, self).__init__()

        # 🔹 Define Discriminator (Binary Classifier for Expert vs. BC Output)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs probability of (s, a) being from expert
        )

        # 🔹 Optimizer and Loss Function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    def forward(self, state, action):
        """ Forward pass to classify (s, a) pairs. """
        x = torch.cat((state, action), dim=-1)  # Concatenate state & action
        return self.model(x)

    def train_discriminator(self, expert_states, expert_actions, bc_states, bc_actions, batch_size=32):
        """ Train Discriminator to classify expert vs. BC-generated data. """
        self.train()
        self.optimizer.zero_grad()

        # Labels: 1 for expert data, 0 for BC policy output
        expert_labels = torch.ones((batch_size, 1))
        bc_labels = torch.zeros((batch_size, 1))

        # Forward pass through Discriminator
        expert_preds = self(expert_states, expert_actions)  # Expert data (should be 1)
        bc_preds = self(bc_states, bc_actions)  # BC policy data (should be 0)

        # Compute Binary Cross-Entropy Loss
        loss_expert = self.criterion(expert_preds, expert_labels)
        loss_bc = self.criterion(bc_preds, bc_labels)
        loss = loss_expert + loss_bc  # Total loss

        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_reward(self, state, action):
        """ Compute discriminator-based reward as -log(D(s, a)) """
        with torch.no_grad():
            reward = -torch.log(self(state, action) + 1e-8)  # Avoid log(0)
        return reward

import torch
import torch.nn as nn
import torch.optim as optim

# class LSTMDiscriminator(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=2, sequence_length=10, lr=1e-3, device="cpu"):
#         super(LSTMDiscriminator, self).__init__()
#
#         self.sequence_length = sequence_length  # Length of state-action sequence
#         self.device=device
#         # 🔹 LSTM Feature Extractor
#         self.lstm = nn.LSTM(input_size=state_dim + action_dim,
#                             hidden_size=hidden_dim,
#                             num_layers=lstm_layers,
#                             batch_first=True)
#
#         # 🔹 Final Classifier (Expert vs. Agent)
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()  # Probability of (s, a) sequence being from expert
#         )
#
#         # 🔹 Optimizer & Loss Function
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#         self.criterion = nn.BCELoss()
#
#     def forward(self, state_action_seq):
#         """
#         Forward pass through LSTM Discriminator.
#         Input: (batch_size, sequence_length, state_dim + action_dim)
#         """
#         lstm_out, _ = self.lstm(state_action_seq)  # Output: (batch, seq_len, hidden_dim)
#         last_hidden = lstm_out[:, -1, :]  # Take last hidden state (batch_size, hidden_dim)
#         return self.classifier(last_hidden)
#
#     def train_discriminator(self, expert_seq, bc_seq, batch_size=32):
#         """ Train Discriminator on sequences of (s, a). """
#         self.train()
#         self.optimizer.zero_grad()
#
#         # Labels: 1 for expert data, 0 for BC-generated data
#         expert_labels = torch.ones((batch_size, 1))
#         bc_labels = torch.zeros((batch_size, 1))
#
#         # Forward pass through Discriminator
#         expert_preds = self(expert_seq)  # Expert (should be 1)
#         bc_preds = self(bc_seq)  # BC (should be 0)
#
#         # Compute Binary Cross-Entropy Loss
#         loss_expert = self.criterion(expert_preds, expert_labels)
#         loss_bc = self.criterion(bc_preds, bc_labels)
#         loss = loss_expert + loss_bc  # Total loss
#
#         # Backpropagation and optimization
#         loss.backward()
#         self.optimizer.step()
#
#         return loss.item()
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lstm_layers=2, sequence_length=10, lr=1e-3, device="cpu"):
        super(LSTMDiscriminator, self).__init__()

        self.sequence_length = sequence_length
        self.device = device  # Store device information

        # 🔹 LSTM Feature Extractor
        self.lstm = nn.LSTM(input_size=state_dim + action_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)

        # 🔹 Final Classifier (Expert vs. Agent)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of (s, a) sequence being from expert
        )

        # 🔹 Optimizer & Loss Function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        # Move model to specified device
        self.to(self.device)

    def forward(self, state_action_seq):
        """
        Forward pass through LSTM Discriminator.
        Input: (batch_size, sequence_length, state_dim + action_dim)
        """
        if not isinstance(state_action_seq, torch.Tensor):
            # Convert to Tensor
            state_action_seq = torch.tensor(state_action_seq, dtype=torch.float32)
        state_action_seq = state_action_seq.to(self.device)  # Move input to device
        lstm_out, _ = self.lstm(state_action_seq)  # Output: (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # Take last hidden state (batch_size, hidden_dim)
        return self.classifier(last_hidden)

    def train_discriminator(self, expert_seq, bc_seq, batch_size=32):
        """ Train Discriminator on sequences of (s, a). """
        self.train()
        self.optimizer.zero_grad()

        # Move input data to device
        expert_seq = expert_seq.to(self.device)
        bc_seq = bc_seq.to(self.device)

        # Labels: 1 for expert data, 0 for BC-generated data
        expert_labels = torch.ones((batch_size, 1), device=self.device)
        bc_labels = torch.zeros((batch_size, 1), device=self.device)

        # Forward pass through Discriminator
        expert_preds = self(expert_seq)  # Expert (should be 1)
        bc_preds = self(bc_seq)  # BC (should be 0)

        # Compute Binary Cross-Entropy Loss
        loss_expert = self.criterion(expert_preds, expert_labels)
        loss_bc = self.criterion(bc_preds, bc_labels)
        loss = loss_expert + loss_bc  # Total loss

        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()