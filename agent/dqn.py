"""
DQN Agent for Vector Observation Learning
Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

# Import Required Packages
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque

from agent.memory import SequentialMemory
from agent.model import QNetwork

# Determine if CPU or GPU computation should be used
from agent.replay_memory import ReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

"""
##################################################
Agent Class
Defines DQN Agent Methods
Agent interacts with and learns from an environment.
"""


class DQNAgent():
    """
    Initialize Agent, inclduing:
        DQN Hyperparameters
        Local and Targat State-Action Policy Networks
        Replay Memory Buffer from Replay Buffer Class (define below)
    """

    def __init__(self, state_size, action_size, dqn_type='DQN', replay_memory_size=1e5, batch_size=64, gamma=0.99,
                 learning_rate=1e-3, target_tau=2e-3, update_rate=4, seed=0):

        """
        DQN Agent Parameters
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            dqn_type (string): can be either 'DQN' for vanillia dqn learning (default) or 'DDQN' for double-DQN.
            replay_memory size (int): size of the replay memory buffer (typically 5e4 to 5e6)
            batch_size (int): size of the memory batch used for model updates (typically 32, 64 or 128)
            gamma (float): paramete for setting the discoun ted value of future rewards (typically .95 to .995)
            learning_rate (float): specifies the rate of model learing (typically 1e-4 to 1e-3))
            seed (int): random seed for initializing training point.
        """
        self.dqn_type = dqn_type
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.memory_frame_stack = 1
        self.buffer_size = int(replay_memory_size)

        """
        # DQN Agent Q-Network
        # For DQN training, two nerual network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stablize learning.
        """
        self.network = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.target_network = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)

        # Replay memory
        self.memory = ReplayMemory(self.action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    ########################################################
    # STEP() method
    #
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    ########################################################
    # ACT() method
    #
    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()
        action_means = torch.mean(action_values.squeeze(-3), 0)
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_means.cpu().data.numpy())
        else:
            print("Random action with epsilon {} probability".format(eps))
            print("action size".format(self.action_size))
            return random.choice(np.arange(self.action_size))

    ########################################################
    # LEARN() method
    # Update value parameters using given batch of experience tuples.
    def learn(self, transition, gamma):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_state, dones = transition
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        # shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            labels_next = self.target_network(predicted_targets).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma * labels_next * (1 - dones))

        loss = criterion(predicted_targets, labels).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    ########################################################
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """

    def soft_update(self, local_model, target_model, tau):
        """
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
