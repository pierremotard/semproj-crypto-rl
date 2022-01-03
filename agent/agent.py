"""
DQN Agent for Vector Observation Learning
Example Developed By:
Michael Richardson, 2018
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
"""

# Import Required Packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

from agent.hierarchical_ppo import ActorCritic

from agent.rollout_buffer import RolloutBuffer

# Determine if CPU or GPU computation should be used
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
Agent interacts with and learns from an environment.
"""


class Agent:
    """
    Initialize Agent, including:
        DQN Hyperparameters
        Local and Targat State-Action Policy Networks
        Replay Memory Buffer from Replay Buffer Class (define below)
    """

    def __init__(self, state_size, action_size, lr_order=0.001, lr_bid=0.001, lr_critic=0.005,
                 gamma=0.99, K_epochs=10, eps_clip=0.2, action_std=0.6, window_size=100):
        """
        Agent Parameters
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            gamma (float): parameter for setting the discount td value of future rewards (typically .95 to .995)
        """
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = gamma
        # TODO: Values of lr to update
        self.lr_actor_order = lr_order
        self.lr_actor_bid = lr_bid
        self.lr_critic = lr_critic
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.window_size = window_size

        """
        # DQN Agent Q-Network
        # For DQN training, two nerual network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stablize learning.
        """

        '''
            Agent is composed of:
                An Actor that takes action, it is a hierarchical model
                    Order network
                    Bid network
                A Critic
        '''

        self.policy = ActorCritic(
            self.state_size, self.action_size, action_std, self.window_size).to(device)

        # Optimize over parameters of both policies, on order and and bid networks of the actor and on the critic network
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.order_net.parameters(),
             'lr': self.lr_actor_order},
            {'params': self.policy.actor.bid_net.parameters(),
             'lr': self.lr_actor_bid},
            {'params': self.policy.critic.parameters(),
             'lr': self.lr_critic}
        ])

        self.MseLoss = nn.MSELoss()

        # Replay memory
        self.memory = RolloutBuffer()

        self.policy_old = ActorCritic(
            self.state_size, self.action_size, action_std, self.window_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.writer = SummaryWriter('runs/hppo')

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ",
                  self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            amount, action_type, action_logprob = self.policy_old.act(state)

        self.memory.states.append(state)
        print("memory in agent  {}".format(self.memory.states[0].shape))
        self.memory.action_types.append(action_type)
        self.memory.amounts.append(amount)
        self.memory.logprobs.append(action_logprob)

        return amount, action_type

    def optimize_model(self, epoch):
        # Monte carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        

        # Normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Check if should keep that but makes sure it's shape Tensor(scalar)
        rewards = torch.squeeze(rewards)

        print("before squeeze {}".format(torch.stack(
            self.memory.states, dim=0).shape))
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(
            self.memory.states, dim=0), dim=0).to(device)
        old_actions = torch.squeeze(torch.stack(
            self.memory.action_types, dim=0), dim=0).to(device)
        old_logprobs = torch.squeeze(torch.stack(
            self.memory.logprobs, dim=0), dim=0).to(device)

        print("- - - - - - - - - - - -")
        for _ in range(self.K_epochs):
            
            print("old states  {}".format(old_states.shape))

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor, Tensor(scalar)
            state_values = torch.squeeze(state_values)
            print(" state values {}".format(state_values.shape))
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # print("ratios : {}".format(ratios))
            # # Should be shape torch.Size([scalar])
            # print("ratios shape : {}".format(ratios.shape))

            # Finding Surrogate Loss
            advantages = rewards - state_values
            # print("advantages : {}".format(advantages))
            # # Should be shape torch.Size([scalar])
            # print("advantages shape : {}".format(advantages.shape))

            # print("rewards : {}".format(rewards))
            # # Should be shape torch.Size([scalar])
            # print("rewards shape : {}".format(rewards.shape))

            # print("state_values : {}".format(state_values))
            # # Should be shape torch.Size([scalar])
            # print("state_values shape : {}".format(state_values.shape))

            # print("loss : {}".format(self.MseLoss(state_values, rewards)))
            # # Should be shape torch.Size([])  i.e. just a scalar
            # print("loss shape : {}".format(self.MseLoss(state_values, rewards).shape))

            # print("dist entropy : {}".format(dist_entropy))
            # # Should be shape torch.Size([scalar])
            # print("dist entropy shape : {}".format(dist_entropy.shape))

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # Should be shape torch.Size([scalar])
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            self.writer.add_scalar(
                "Loss/train", loss.mean(), global_step=epoch)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear memory after all epochs
        self.memory.clear()

        self.writer.flush()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path))
        self.policy.load_state_dict(torch.load(checkpoint_path))
