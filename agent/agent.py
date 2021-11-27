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

from agent.model import DQN

# Determine if CPU or GPU computation should be used
from agent.replay_memory import ReplayMemory, Transition

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


class Agent():
    """
    Initialize Agent, inclduing:
        DQN Hyperparameters
        Local and Targat State-Action Policy Networks
        Replay Memory Buffer from Replay Buffer Class (define below)
    """

    def __init__(self, state_size, action_size, replay_memory_size=10000, batch_size=64, gamma=0.99,
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
        self.state_size = state_size
        self.action_size = action_size
    
        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.memory_frame_stack = 1
        

        """
        # DQN Agent Q-Network
        # For DQN training, two nerual network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stablize learning.
        """
        self.policy_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net = DQN(self.state_size, self.action_size).to(device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learn_rate)

        # Replay memory
        self.memory = ReplayMemory(replay_memory_size)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

  
    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if random.random() < eps:
            print("Random action with epsilon {} probability".format(eps))
            print("action size".format(self.action_size))
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
            #return random.choice(np.arange(self.action_size))

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1)[1].view(1, 1)

        #action_means = torch.mean(action_values.squeeze(-3), 0)
        #return np.argmax(action_means.cpu().data.numpy())

    def push_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
