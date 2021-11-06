from configurations import LOGGER
import os
import gym
import gym_trading
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class Agent(object):
    name = 'DQN'

    def __init__(self, number_of_training_steps=1e5, gamma=0.999, load_weights=False, training=True,
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

        # Create log dir
        log_dir = "tmp/"
        os.makedirs(log_dir, exist_ok=True)

        # Create environment
        self.env = gym.make(**kwargs)
        self.env = Monitor(self.env, log_dir)
        self.eval_env = gym.make(**kwargs)
        self.env_name = self.env.env.id

        # Create agent
        # NOTE: 'Keras-RL' uses its own frame-stacker
        self.memory_frame_stack = 1  # Number of frames to stack e.g., 1.

        #Instantiate DQN model
        self.model = DQN("MlpPolicy", self.env, verbose=1)

        self.train = True
        self.cwd = os.path.dirname(os.path.realpath(__file__))

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
            self.env_name, "dqn")
        weights_filename = os.path.join(output_directory, weight_name)
        LOGGER.info("weights_filename: {}".format(weights_filename))

        if self.load_weights:
            LOGGER.info('...loading weights for {} from\n{}'.format(
                self.env_name, weights_filename))
            self.agent.load_weights(weights_filename)

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

            #Train the agent
            self.model.learn(total_timesteps=int(25))
            #Save the agent
            self.model.save("dqn_trained_saved")
            del self.model

            LOGGER.info("training over.")
        else:
            LOGGER.info('Starting TEST...')
            self.model = DQN.load("dqn_trained_saved")

            # Evaluate the trained agent
            mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=3, deterministic=True)

            print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")