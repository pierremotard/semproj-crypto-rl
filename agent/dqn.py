# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Conv2D
# from keras.optimizers import Adam
# from rl.agents.dqn import DQNAgent
# from rl.memory import SequentialMemory
# from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from agent.Hierarchical_RL_Attention import ActorNetwork, OrderNetwork, ReplayBuffer, compute_td_loss
from stable_baselines3 import PPO, DQN, DDPG, A2C, TD3, SAC
from configurations import LOGGER
import os
import gym
import gym_trading
import torch
import torch.optim as optim
import numpy as np
import math

class Agent(object):
    name = 'DQN'

    def __init__(self, number_of_training_steps=1e5, training=True, gamma=0.999, load_weights=False,
                 visualize=False, dueling_network=True, double_dqn=True, nn_type='mlp',
                 **kwargs):
        """
        Agent constructor
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

        # Create environment
        self.env = gym.make(**kwargs)
        self.env_name = self.env.env.id

        # Create agent
        # NOTE: 'Keras-RL' uses its own frame-stacker
        self.memory_frame_stack = 1  # Number of frames to stack e.g., 1.
        # self.model = self.create_model(name=self.neural_network_type)
        # self.memory = SequentialMemory(limit=10000,
        #                               window_length=self.memory_frame_stack)
        self.train = True #kwargs["training"] #self.env.env.training
        self.cwd = os.path.dirname(os.path.realpath(__file__))

        # create the agent
        # self.agent = A2C('MlpPolicy', self.env, verbose=1, tensorboard_log="./a2c/")
        # self.agent = A2C('CnnPolicy', self.env, verbose=1, tensorboard_log="./a2c/")
        # self.agent = PPO('MlpPolicy', self.env, verbose=1, tensorboard_log="./ppo/")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.replay_initial = 100
        self.replay_buffer = ReplayBuffer(1000)

        self.state_dim_act = self.env.observation_space.shape[1]
        self.state_dim_order = self.env.observation_space.shape[1]

        # self.action_dim = 2
        self.action_dim = 3
        self.batch_size = 8
        self.hidden_dim = 64

        self.policy_lr = 1e-3
        self.policy_net_act = ActorNetwork(self.state_dim_act,self.hidden_dim)
        self.optimizer_act = optim.Adam(self.policy_net_act.parameters(),lr=self.policy_lr)

        self.policy_net_order = OrderNetwork(self.state_dim_order,self.action_dim,self.hidden_dim)
        self.optimizer_order = optim.Adam(self.policy_net_order.parameters(),lr=self.policy_lr)


    def __str__(self):
        # msg = '\n'
        # return msg.join(['{}={}'.format(k, v) for k, v in self.__dict__.items()])
        return 'Agent = {} | env = {} | number_of_training_steps = {}'.format(
            Agent.name, self.env_name, self.number_of_training_steps)

   
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
            self.env_name, self.neural_network_type)
        weights_filename = os.path.join(output_directory, weight_name)
        LOGGER.info("weights_filename: {}".format(weights_filename))

        if self.load_weights==True:
            LOGGER.info('...loading weights for {} from\n{}'.format(
                self.env_name, weights_filename))
            print("Loading weights.")
            # self.agent.load_weights(weights_filename)
            self.policy_net_act.load_state_dict(torch.load("policy_net_act_weights.pth"))
            self.optimizer_act.load_state_dict(torch.load("optimizer_act_weights.pth"))
            self.policy_net_order.load_state_dict(torch.load("policy_net_order_weights.pth"))
            self.optimizer_order.load_state_dict(torch.load("optimizer_order_weights.pth"))
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

            # callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
            #                                     interval=250000)]
            # callbacks += [FileLogger(log_filename, interval=100)]

            num_frames = 10000  # Steps
            weights_act = []
            weights_ord = []
            rewards = []
            loss_order = []
            loss_actor = []
            loss = []
            profit = []
            forecast = []
            action_list = []
            gamma = 0.99

            epsilon_start = 0.90
            epsilon_final = 0.01
            epsilon_decay = 10000
            epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
                -1. * frame_idx / epsilon_decay)        

            LOGGER.info('Starting training...')
            #self.agent.learn(total_timesteps=self.number_of_training_steps)

            state = self.env.reset()

            for step in range(self.number_of_training_steps):
                state = torch.FloatTensor(state)
                epsilon = epsilon_by_frame(step)
                state = torch.Tensor(state).unsqueeze(dim=0)
                act, w_act = self.policy_net_act.act(state, epsilon)
                forecast.append(act)

                # state_order = torch.cat([state, torch.unsqueeze(act.transpose(-2, -1), 0)], 1)
                state_order = torch.cat([state, act.unsqueeze(0)], 1)

                amount, side, w_ord = self.policy_net_order.act(state_order, epsilon)  # for amount and action prediction
                action_list.append(side)
                if torch.is_tensor(side):
                    side = int(side.cpu().detach().numpy()[0])
                if torch.is_tensor(amount):
                    amount = float(amount.cpu().detach().numpy())
                # print(type(dec), type(order))
                # print(type([dec, order]))
                action = torch.tensor([amount, side])
                # action = np.expand_dims(action, axis=1)
                # action = action.T

                # print(action)
                # next_state, reward, done, _ = self.env.step(action)

                next_state, reward, done, _ = self.env.step(action)

                # print("This is reward outside: ", type(reward), reward)
                # bal, net, shares_held, prof = self.env.render()

                self.replay_buffer.push(state, state_order, act, side, reward, next_state, done)

                step += 1
                state = next_state
                # print(rewards)
                if len(self.replay_buffer) > self.replay_initial:
                    # print(len(replay_buffer), "skr skr")
                    ord_l, act_l, TD_Loss = compute_td_loss(
                        self.batch_size, self.replay_buffer, self.device, self.policy_net_act, self.policy_net_order,
                        self.optimizer_order, gamma, rewards, self.optimizer_act)
                    
                    loss_actor.append(act_l), loss_order.append(ord_l), loss.append(TD_Loss)

                if (step % 1000) == 0:
                    print("Step : {}".format(step))
                #     weights_act.append(w_act)
                #     weights_ord.append(w_ord)
                #     print('Step-', str(step), '/', str(num_frames), '| Profit-', net, '| Model Loss-', ord_l)
                #     torch.save(
                #         {'model_state_dict': self.policy_net_act.state_dict(),
                #         'optimizer_state_dict_act': self.optimizer_act.state_dict(),
                #         'loss': TD_Loss}, checkpoint_name + '/policy_net_act.pth.tar')  # save PolicyNet
                #     torch.save(
                #         {'model_state_dict': self.policy_net_order.state_dict(),
                #         'optimizer_state_dict_order': self.optimizer_order.state_dict(),
                #         'loss': TD_Loss}, checkpoint_name + '/policy_net_order.pth.tar')  # save PolicyNet

                rewards.append(reward), 
                # loss_actor.append(act_l), loss_order.append(ord_l), loss.append(TD_Loss)

                if done:
                    state = self.env.reset()



            LOGGER.info("training over.")
            LOGGER.info('Saving AGENT weights...')
            torch.save(self.policy_net_act.state_dict(), "policy_net_act_weights.pth")
            torch.save(self.optimizer_act.state_dict(), "optimizer_act_weights.pth")
            torch.save(self.policy_net_order.state_dict(), "policy_net_order_weights.pth")
            torch.save(self.optimizer_order.state_dict(), "optimizer_order_weights.pth")


            LOGGER.info("AGENT weights saved.")
        else:
            LOGGER.info('Starting TEST...')

            self.policy_net_act.load_state_dict(torch.load("policy_net_act_weights.pth"))
            self.optimizer_act.load_state_dict(torch.load("optimizer_act_weights.pth"))

            self.policy_net_order.load_state_dict(torch.load("policy_net_order_weights.pth"))
            self.optimizer_order.load_state_dict(torch.load("optimizer_order_weights.pth"))


            state = self.env.reset()
            rewards = []
            for step in range(15000):
                state = torch.FloatTensor(state)
                state = torch.Tensor(state).unsqueeze(dim=0)
                act, w_act = self.policy_net_act.act(state, 0)
                # state_order = torch.cat([state, torch.unsqueeze(act.transpose(-2, -1), 0)], 1)
                state_order = torch.cat([state, act.unsqueeze(0)], 1)

                amount, side, w_ord = self.policy_net_order.act(state_order, 0)
                if torch.is_tensor(side):
                    side = int(side.cpu().detach().numpy()[0])
                if torch.is_tensor(amount):
                    amount = float(amount.cpu().detach().numpy())

                # action = np.array([side, amount])
                # action = np.expand_dims(action, axis=1)
                # action = action.T
                action = torch.tensor([amount, side])
                # print(action)
                next_state, reward, done, info = self.env.step(action)
                # print("This is reward outside: ", type(reward), reward)

                rewards.append(reward)

                if done:
                    print("Breaks episode.")
                    break

                state = next_state

                if step % 500 == 0:
                    print(f"Step: {step}")
            

            self.env.plot_observation_history("plot_ep_history_ahrl")
            self.env.plot_trade_history("plot_trades_history_ahrl")
            self.env.portfolio.get_portfolio()
