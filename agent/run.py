from collections import deque
from comet_ml import Experiment
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_days = ["XBTUSD_2020-01-02.csv.xz", "XBTUSD_2020-01-03.csv.xz", "XBTUSD_2020-01-04.csv.xz", 
            "XBTUSD_2020-01-05.csv.xz", "XBTUSD_2020-01-06.csv.xz", "XBTUSD_2020-01-07.csv.xz", "XBTUSD_2020-01-08.csv.xz",
            "XBTUSD_2020-01-09.csv.xz", "XBTUSD_2020-01-10.csv.xz", "XBTUSD_2020-01-11.csv.xz", "XBTUSD_2020-01-12.csv.xz"]

#----------- Env hyperparameters -----------#


#----------- PPO hyperparameters -----------#
max_episode_len = 15000
update_timestep = 80 # max_episode_len * 2       # update policy every n timesteps
# update policy for K epochs in one PPO update
K_epochs = 50

eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_order = 0.001            # learning rate for order actor network
lr_bid = 0.001              # learning rate for bid actor network
lr_critic = 0.005           # learning rate for critic network

# starting std for action distribution (Multivariate Normal)
action_std = 0.6
# action_std decay frequency (in num timesteps)
action_std_decay_freq = int(2e5)
# linearly decay action_std (action_std = action_std - action_std_decay_rate)
action_std_decay_rate = 0.05
# minimum action_std (stop decay after action_std <= min_action_std)
min_action_std = 0.1

save_model_every = 1000
id_trained_model = 0
checkpoint_path = "saved_models/PPO_{}.pth".format(id_trained_model)

random_seed = 0         # set random seed if required (0 = no random seed)

hyper_params = {
    "lr_order": 0.001,
    "lr_bid": 0.001,
    "lr_critic": 0.05
}


class Run(object):
    name = 'HPPO'

    def __init__(self, mode, nb_training_days=3, nb_testing_days=2, number_of_training_steps=1e2, gamma=0.999, load_weights=False,
                 visualize=False, dueling_network=True, double_dqn=True, logger=False,
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
        self.load_weights = load_weights
        self.number_of_training_steps = number_of_training_steps
        self.visualize = visualize

        # Create log dir
        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)

        self.use_logger = logger
        if self.use_logger:
            self.experiment = Experiment(
                project_name="crypto-trading", api_key="aASgA0tUpW7Y4FxbuD0egr84z", workspace="pierremotard")
            self.experiment.log_parameters(hyper_params)

        # Create environment
        self.env = gym.make(**kwargs)
        self.eval_env = gym.make(**kwargs)
        self.env_name = self.env.env.id

        self.kwargs = kwargs

        # Create agent
        # NOTE: 'Keras-RL' uses its own frame-stacker
        self.memory_frame_stack = 1  # Number of frames to stack e.g., 1.

        self.daily_returns = []

        # Instantiate DQN model
        print(self.env.observation_space.shape)
        print(self.env.observation_space.shape[1])
        print(self.env.action_space.n)
        self.window_size = self.env.window_size
        self.agent = Agent(self.env.observation_space.shape[1], action_size=self.env.action_space.n,
                           lr_order=lr_order, lr_bid=lr_bid, lr_critic=lr_critic, K_epochs=K_epochs, action_std=action_std, window_size=self.window_size)

        self.train = str(mode) == 'train'
        print("Mode TRAINING") if self.train else print("Mode TESTING")
        
        self.nb_training_days = nb_training_days
        self.nb_testing_days = nb_testing_days

            
        self.cwd = os.path.dirname(os.path.realpath(__file__))

    def __str__(self):
        # msg = '\n'
        # return msg.join(['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return 'Agent = {} | env = {} | number_of_training_steps = {}'.format(
            Run.name, self.env_name, self.number_of_training_steps)

    def train_agent(self, nb_episodes=1, eps_start=1.0, eps_end=0.01, eps_decay=0.996):
        """
            Params
            ======
                nb_episodes (int): maximum number of training epsiodes
                eps_start (float): starting value of epsilon, for epsilon-greedy action selection
                eps_end (float): minimum value of epsilon
                eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon

        """

        steps_buy = []
        steps_sell = []

        timestep = 0
        while timestep <= nb_episodes:
            print("------------------------------------")
            print("New episode {}".format(timestep))
            print("------------------------------------\n\n")
            state = self.env.reset()
            print("Init state shape {}".format(state.shape))
            current_episode_reward = 0

            for episode_step in range(max_episode_len):
                if episode_step % 500:
                    print("New step {}".format(episode_step))

                state = torch.Tensor(state).unsqueeze(0)
                print("state in run {}".format(state.shape))

                amount, action_type = self.agent.act(state)


                state, reward, done, _ = self.env.step(
                    torch.cat((amount, action_type.unsqueeze(0)), dim=0))

                # next_state = torch.tensor(next_state, device=device)
                # reward = torch.tensor([reward], device=device)
                if self.use_logger:
                    self.experiment.log_metric(
                        "reward", reward, step=episode_step*(timestep+1))

                self.agent.memory.rewards.append(reward)

                current_episode_reward += reward

                # Update agent by optimizing its model
                if episode_step % update_timestep == 0:
                    self.agent.optimize_model(episode_step)

                # TODO: If continuous action space, add decay action std
                if episode_step % action_std_decay_freq == 0:
                    self.agent.decay_action_std(
                        action_std_decay_rate, min_action_std)

                # TODO: Save the model checkpoint
                LOGGER.info("Save model checkpoints...")
                id_trained_model = 7
                if episode_step % save_model_every == 0:
                    self.agent.save(checkpoint_path)

                if done:
                    break
            timestep += 1
        
        self.env.close()
        self.agent.writer.flush()
        self.agent.writer.close()

        torch.save(self.agent.policy.actor.order_net.state_dict(),
                   "saved_models/order_net_checkpoint.pth")
        torch.save(self.agent.policy.actor.bid_net.state_dict(),
                   "saved_models/bid_net_checkpoint.pth")

    def test_agent(self):
        print("Load network from checkpoint ...")
        self.agent.policy.actor.order_net.load_state_dict(
            torch.load("saved_models/order_net_checkpoint.pth"))
        self.agent.policy.actor.bid_net.load_state_dict(
            torch.load("saved_models/bid_net_checkpoint.pth"))
        print("Checkpoint loaded.")
        print("Start testing ...")

        print("env details")
        self.log_environment_details()

        state_size = self.env.observation_space.shape[0]
        # action_size = env_test.action_space["action_type"].n

        # test_agent = Agent(state_size, 3, lr_order, lr_bid,
        #                 lr_critic, gamma, K_epochs, eps_clip, action_std)

        self.agent.load(checkpoint_path)

        episode_reward = 0
        state = self.env.reset()
        
        info = []
        buys = []
        sells = []


        for i in range(600):
            state = torch.Tensor(state)
            action = self.agent.act(state)


            if action[1].item() == 0:
                buys.append(i)
            if action[1].item() == 1:
                sells.append(i)

            state, reward, done, info = self.env.step(action)
            episode_reward += reward
            # self.env.render()    not working yet

            if done:
                print("Reaches end of data or no more CA$H :(")
                break

                self.env.get_transaction_df(i_episode)
        
        self.daily_returns.append(self.env.portfolio.get_net_worth())

    def start(self) -> None:
        """
        Entry point for agent training and testing
        :return: (void)
        """
        output_directory = os.path.join(self.cwd, 'hppo_weights')
        if not os.path.exists(output_directory):
            LOGGER.info(
                '{} does not exist. Creating Directory.'.format(output_directory))
            os.mkdir(output_directory)

        weight_name = 'hppo_{}_{}_weights.h5f'.format(
            self.env_name, "hppo")
        weights_filename = os.path.join(output_directory, weight_name)
        LOGGER.info("weights_filename: {}".format(weights_filename))
        if self.train:
            print("Starts training.")
            # Train the agent
            for i in range(1, self.nb_training_days):
                if self.use_logger:
                    with self.experiment.train():
                        self.train_agent(nb_episodes=1)
                else:
                    self.train_agent(nb_episodes=1)

                self.env.get_transaction_df(data_days[i], "run_nb_{}".format(i))
                if i == self.nb_training_days-1:
                    break
                self.kwargs["fitting_file"] = data_days[i]
                self.kwargs["testing_file"] = data_days[i+1]
                self.env = gym.make(**self.kwargs)
            print(" ----- ")
            LOGGER.info("training over.")
            print(self.daily_returns)

        else:
            print("Starts testing.")
            for i in range(1, self.nb_testing_days):
                if self.use_logger:
                    with self.experiment.test():
                        self.test_agent()
                else:
                    self.test_agent()

                self.env.get_transaction_df(data_days[i], "run_nb_{}".format(i))
                self.kwargs["fitting_file"] = data_days[i]
                self.kwargs["testing_file"] = data_days[i+1]
                self.env = gym.make(**self.kwargs)
            
            print("Finish testing.")
            
            print(self.env.position_stats())
            print(self.daily_returns)
            self.env.plot_trade_history("plots_viz")

        self.env.portfolio.get_portfolio()
        print("buys -> {}".format(self.env.buys))
        print()
        print("sells -> {}".format(self.env.sells))
        self.env.close()

    def log_environment_details(self):
        print("-- Environment details -- ")
        print(self.env_name)
        print(self.env.observation_space.shape)
