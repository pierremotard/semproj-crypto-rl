from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import torch

class RolloutBuffer(ABC):
    def __init__(self,
    buffer_size,
    gae_lambda: float = 1,
    gamma: float = 0.99,
    device: Union[torch.device, str] = "cpu") -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.device = device
        self.states, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.values, self.log_probs = None, None, None

        self.pos = 0
        self.full = False

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def reset(self) -> None:
        """
        Reset buffer
        """ 
        self.states = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    