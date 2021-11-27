from collections import deque

import torch

from agent.agent import Agent
from configurations import LOGGER
import os
import gym
import gym_trading
from stable_baselines3 import DQN
#from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.monitor import Monitor
import numpy as np

WINDOW_SIZE = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Run(object):
    name = 'DQN'

    def __init__(self, number_of_training_steps=1e5, gamma=0.999, load_weights=False, training=True,
                 visualize=False, dueling_network=True, double_dqn=True, nn_type='mlp',
                 **kwargs):
        """
        Run constructor
        :param window_size: int, number of lags to include in observation
        :param max_position: int, maximum number of positions able to be held in inventory
        :param fitting_file: str, file used for z-score fitting
        :param testing_file: str,file used for dqn experiment
        :param env: environment name
        :param seed: int, random seed number
        :param action_repeats: int, number of steps to take in environment between actions
        :param number_of_training_steps: int, number of steps to train agent for
        :param gamma: float, value between 0 and 1 used to discount future DQN returns
        :param format_3d: boolean, format observation as matrix or tensor
        :param train: boolean, train or test agent
        :param load_weights: boolean, import existing weights
        :param z_score: boolean, standardize observation space
        :param visualize: boolean, visualize environment
        :param dueling_network: boolean, use dueling network architecture
        :param double_dqn: boolean, use double DQN for Q-value approximation
        """
        # Agent arguments
        # self.env_name = id
        self.neural_network_type = nn_type
        self.load_weights = load_weights
        self.number_of_training_steps = number_of_training_steps
        self.visualize = visualize

        # Create log dir
        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)

        # Create environment
        self.env = gym.make(**kwargs)
        self.eval_env = gym.make(**kwargs)
        self.env_name = self.env.env.id

        # Create agent
        # NOTE: 'Keras-RL' uses its own frame-stacker
        self.memory_frame_stack = 1  # Number of frames to stack e.g., 1.

        # Instantiate DQN model
        print(self.env.observation_space.shape)
        print(self.env.observation_space.shape[1])
        print(self.env.action_space.n)
        self.agent = Agent(self.env.observation_space.shape[1], action_size=self.env.action_space.n)

        self.train = True
        self.cwd = os.path.dirname(os.path.realpath(__file__))

    def __str__(self):
        # msg = '\n'
        # return msg.join(['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return 'Agent = {} | env = {} | number_of_training_steps = {}'.format(
            Run.name, self.env_name, self.number_of_training_steps)

    def train_agent(self, nb_episodes=100, max_t=10, eps_start=1.0, eps_end=0.01, eps_decay=0.996):
        """
            Params
            ======
                nb_episodes (int): maximum number of training epsiodes
                max_t (int): maximum number of timesteps per episode
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon

        """
        scores = []
        scores_window = deque(maxlen=100)
        eps = eps_start
        for i_episode in range(nb_episodes):
            state = self.env.reset()
            state = torch.Tensor(state)
            score = 0
            for t in range(max_t):

                action = self.agent.act(state, eps)
                print("ACTION DECIDED {}".format(action.item()))
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state, device=device)
                reward = torch.tensor([reward], device=device)
                if done:
                    break

                self.agent.push_memory(state, action, next_state, reward)

                state = next_state
                

                self.agent.optimize_model()

                if done:
                    break

                score += reward
                scores_window.append(score)
                scores.append(score)
                eps = max(eps * eps_decay, eps_end)
                #print('\rEpisode {}\tAverage Score {:.2f}'.format(i_episode, np.mean(scores_window)), end=" ")

                #if np.mean(scores_window) >= 100:
                #print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(i_episode - 100,
                #                                                                            np.mean(scores_window)))
                #torch.save(self.agent.policy_net.state_dict(), 'checkpoint.pth')

        self.env.close()
        return scores

    def start(self) -> None:
        """
        Entry point for agent training and testing
        :return: (void)
        """
        output_directory = os.path.join(self.cwd, 'dqn_weights')
        if not os.path.exists(output_directory):
            LOGGER.info('{} does not exist. Creating Directory.'.format(output_directory))
            os.mkdir(output_directory)

        weight_name = 'dqn_{}_{}_weights.h5f'.format(
            self.env_name, "dqn")
        weights_filename = os.path.join(output_directory, weight_name)
        LOGGER.info("weights_filename: {}".format(weights_filename))

        if self.train:

            # Train the agent
            self.train_agent()
            print(" ----- ")
            LOGGER.info("training over.")
        else:
            print("Load network from checkpoint ...")
            self.agent.network.load_state_dict(torch.load("checkpoint.pth"))
            print("Checkpoint loaded.")
            print("Start testing ...")
            for i in range(2):
                state = self.env.reset()
                for j in range(3):
                    action = self.agent.act(state)
                    state, reward, done, _ = self.env.step(action)
                    print("Reward of test is {}".format(reward))
                    if done:
                        break

            print("Finish testing.")
            self.env.get_transaction_df()
            print(self.env.position_stats())
            self.env.close()
