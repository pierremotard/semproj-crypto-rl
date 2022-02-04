import gym
import json
import math
import numpy as np
import pandas as pd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Torch and Baseline Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_norm_
from stable_baselines3.common.vec_env import DummyVecEnv


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


# Attention module
class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype=torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.
        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)
            The output must be a softmax weighting over the seq_len annotations.
        """

        # ------------
        # FILL THIS IN
        # ------------
        batch_size, seq_len, hidden_size = keys.size()
        q = self.Q(queries.view(batch_size, -1, hidden_size))  # .view(-1, hidden_size)
        k = self.K(keys)  # .view(-1, hidden_size)
        v = self.V(values)
        unnormalized_attention = torch.bmm(k, q.transpose(2, 1)) * self.scaling_factor
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights.transpose(2, 1), v)
        return context, attention_weights


# Hierarchical model
class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(ActorNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=self.hidden_size, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.attn = ScaledDotAttention(2 * self.hidden_size)
        self.fc = nn.Linear(2 * self.hidden_size, 151) # Change to 153 if added networth and balance in position_features

    def forward(self, state):
        h0 = torch.zeros(2, state.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2, state.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(state, (h0, c0))
        out, w_act = self.attn(out, out, out)
        out = self.fc(out[:, -1, :])
        # print("Into forward: ", out.shape)
        return out, w_act

    def act(self, state, epsilon):
        pred, w_act = self.forward(state)
        return pred, w_act


class OrderNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(OrderNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=self.hidden_size, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.attn = ScaledDotAttention(2 * self.hidden_size)
        self.fc = nn.Linear(2 * self.hidden_size, self.num_actions)

    def forward(self, state):
        h0 = torch.zeros(2, state.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2, state.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(state, (h0, c0))
        out, w_ord = self.attn(out, out, out)
        out = self.fc(out[:, -1, :])
        out = F.softmax(out)
        prob = max(out[0, :])
        action = torch.argmax(out, dim=1)
        return out, prob, action, w_ord

    def act(self, state, epsilon):
        if random.random() > epsilon:
            _, prob, action, w_ord = self.forward(state)
            # print("Action from forward {}".format(action))
        else:
            prob = random.uniform(0, 1)
            action = random.randint(0, 2)
            w_ord = []
        return prob, action, w_ord


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, state_order, act, dec, reward, next_state, done):
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        state = state.detach().cpu().numpy()
        next_state = next_state
        state_order = state_order.detach().cpu().numpy()
        act = act.detach().cpu().numpy()
        self.buffer.append((state, state_order, act, dec, reward, next_state, done))

    def sample(self, batch_size):
        state, state_order, act, dec, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), np.concatenate(state_order), np.concatenate(act), dec, reward, np.concatenate(
            next_state), done

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(batch_size, replay_buffer, device, policy_net_act, policy_net_order,
                    optimizer_order, gamma, rewards, optimizer_act):
    state, state_order, act, dec, reward, next_state, done = replay_buffer.sample(batch_size)

    action_act = torch.LongTensor(act).to(device)
    state = torch.FloatTensor(np.float32(state))#.to(device)
    state_order = torch.FloatTensor(state_order).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    reward = torch.FloatTensor(np.array(reward)).to(device)
    done = torch.FloatTensor(np.array(done)).to(device)

    criterion = nn.SmoothL1Loss(reduction='none')

    next_state = torch.Tensor(state).view(batch_size, -1, state.shape[-1]).to(device)
    
    state = state.to(device)
    # Manager(Actor) Network
    q_values_act, _ = policy_net_act(state)
    # print(q_values_act.shape)
    # print(action_act.shape)

    

    next_q_values_act, _ = policy_net_act(next_state)

    torch.clamp(action_act, min=0)
    q_value_act = q_values_act.gather(1, action_act).squeeze(1)
    next_q_value_act = next_q_values_act  # .max(1)[0]

    norm = max(rewards) if rewards != [] else 1
    norm = torch.FloatTensor(np.array(norm)).to(device)

    # expected_q_value_act = (reward / norm) + gamma * next_q_value_act * (1 - done)
    expected_q_value_act = (reward / norm).unsqueeze(1) + gamma * next_q_value_act * (1 - done.unsqueeze(1))

    act_loss = criterion(q_value_act, Variable(expected_q_value_act.data))
    act_loss = torch.sum(torch.mean(act_loss, 0))
    #     act_loss = (q_value_act - Variable(expected_q_value_act.data)).pow(2).mean()
    #     optimizer_act.zero_grad()
    #     act_loss.backward()
    #     optimizer_act.step()

    # Order Network
    q_values_order, _, _, _ = policy_net_order(state_order)
    # print(q_value_act.shape)
    # q_value_act = torch.unsqueeze(q_value_act, 0).permute(1, 2, 0)
    q_value_act = torch.unsqueeze(q_value_act, 0).permute(1, 0, 2)
    # print(next_state.shape)
    # print(q_value_act.shape)
    next_state_order = torch.cat([next_state, q_value_act], 1).to(device)
    next_q_values_order, _, _, _ = policy_net_order(next_state_order)
    next_q_value_order = next_q_values_order.unsqueeze(1).max(1)[0].type(torch.FloatTensor).to(device)

    # expected_q_value_order = (reward / norm) + gamma * next_q_value_order * (1 - done)
    expected_q_value_order = (reward / norm).unsqueeze(1) + gamma * next_q_value_order * (1 - done.unsqueeze(1))

    order_loss = criterion(q_values_order, Variable(expected_q_value_order.data))
    order_loss = torch.sum(torch.mean(order_loss, 0))
    #     order_loss = (torch.FloatTensor(q_values_order) - torch.FloatTensor(expected_q_value_order)).pow(2).mean()
    total_loss = 0.5 * act_loss + 0.5 * order_loss
    optimizer_order.zero_grad()
    optimizer_act.zero_grad()
    total_loss.backward()
    clip_grad_norm_(policy_net_act.parameters(), 2)
    clip_grad_norm_(policy_net_order.parameters(), 2)
    optimizer_order.step()
    optimizer_act.step()

    #     total_loss = 0.3*act_loss + 0.7*order_loss

    return order_loss.detach().cpu().numpy(), act_loss.detach().cpu().numpy(), total_loss.detach().cpu().numpy()
