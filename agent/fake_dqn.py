from agent.dqnagent import DQNAgent
from agent.model import QNetwork
from configurations import LOGGER
import os
import gym
import gym_trading
import pylab
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count

from agent.replay_memory import ReplayMemory

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, number_of_training_steps=1e5, gamma=0.999, load_weights=False,
                 visualize=False, dueling_network=True, double_dqn=True, nn_type='mlp',
                 **kwargs):
        self.env = gym.make(**kwargs)
        self.env_name = self.env.env.id
        self.load_weights = load_weights
        self.number_of_training_steps = number_of_training_steps
        self.visualize = visualize


        features_shape = (self.memory_frame_stack, *self.env.observation_space.shape)
        self.model = QNetwork(state_size=features_shape, action_size=self.env.action_space.n, seed=23)


        self.train = self.env.env.training
        self.cwd = os.path.dirname(os.path.realpath(__file__))

        self.agent = DQNAgent(
            dqn_type='DDQN',
            state_size=features_shape,
            action_size=self.env.action_space.n,
        )

    def __str__(self):
        # msg = '\n'
        # return msg.join(['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return 'Agent = {} | env = {} | number_of_training_steps = {}'.format(
            Agent.name, self.env_name, self.number_of_training_steps)


    def start(self):
        """
            Entry point for agent training and testing
            :return: (void)
        """
        output_directory = os.path.join(self.cwd, 'dqn_weights')
        if not os.path.exists(output_directory):
            LOGGER.info('{} does not exist. Creating Directory.'.format(output_directory))
            os.mkdir(output_directory)

        weight_name = 'dqn_{}_{}_weights.h5f'.format(
            self.env_name, 'ddqn')
        weights_filename = os.path.join(output_directory, weight_name)
        LOGGER.info("weights_filename: {}".format(weights_filename))

        """
        if self.load_weights:
            LOGGER.info('...loading weights for {} from\n{}'.format(
                self.env_name, weights_filename))
            self.agent.load_weights(weights_filename)
        """

        if self.train:
            step_chkpt = '{step}.h5f'
            step_chkpt = 'dqn_{}_weights_{}'.format(self.env_name, step_chkpt)
            checkpoint_weights_filename = os.path.join(self.cwd,
                                                       'dqn_weights',
                                                       step_chkpt)
            LOGGER.info("checkpoint_weights_filename: {}".format(
                checkpoint_weights_filename))
            log_filename = os.path.join(self.cwd, 'dqn_weights',
                                        'dqn_{}_log.json'.format(self.env_name))
            LOGGER.info('log_filename: {}'.format(log_filename))

            LOGGER.info('Starting training...')
            # loop from num_episodes
            num_episodes = 50

            for i_episode in range(num_episodes):
                # Initialize the environment and state
                state = self.env.reset() #TODO
                # Select and perform an action
                action = self.agent.act(state)
                action, reward, new_state, state, done = self.env.step(action, i_episode)  #.item() on action
                if done:
                    break

                # Store the transition in memory
                self.agent.store(state, action, new_state, reward)
                self.agent.optimize_model()


            print('Complete')
            self.env.close()

            #mean_reward, std_reward = evaluate_policy(self.policy, self.env, n_eval_episodes=10, deterministic=True)

            #print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")