import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OrderNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(OrderNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
            nn.Sigmoid()
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)

        #  shape [batch_size, nb_features], NOT [batch_size, nb_features, 1]
        return self.layers(x.view(x.size(0), -1))


class BidNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(BidNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)  # check if softmax because not in paper
        )

    def forward(self, x):
        x = x.to(device)
        #  shape [batch_size, nb_features], NOT [batch_size, nb_features, 1]
        return self.layers(x.view(x.size(0), -1))


class Actor(nn.Module):
    def __init__(self, state_size, action_std_init=0.6):
        super(Actor, self).__init__()
        self.order_net = OrderNetwork(state_size, 1)
        self.bid_net = BidNetwork(state_size+1, 3)

        self.action_var = torch.full(
            (1,), action_std_init * action_std_init).to(device)

    def act(self, state):

        amount_order_net = self.order_net(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(amount_order_net, cov_mat)
        amount = torch.clip(dist.sample(), min=0, max=1)
        amount_logprob = dist.log_prob(amount)  # maybe to use later

        print("Amount sampled : {}".format(amount))

        # Bid network takes as input [observed state, sampled amount]
        state_with_amount = torch.cat((state, amount), dim=1)

        out_bid_net = self.bid_net(state_with_amount)

        return amount, out_bid_net


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, action_std_init):
        super(ActorCritic, self).__init__()

        self.action_size = action_size

        self.actor = Actor(state_size, action_std_init)

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size+1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full(
                (1,), new_action_std * new_action_std).to(device)
        else:
            print(
                "--------------------------------------------------------------------------------------------")
            print(
                "WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print(
                "--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        amount, action_type_probs = self.actor.act(
            state)

        dist = Categorical(action_type_probs)
        action_type = dist.sample()
        action_logprob = dist.log_prob(action_type)

        return amount, action_type, action_logprob

    def evaluate(self, state, action):
        action_actor = self.actor.act(state)
        amount = action_actor[0]
        action_type_probs = action_actor[1]
        dist = Categorical(action_type_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_with_amount = torch.cat((state, amount), dim=1)
        state_values = self.critic(state_with_amount)

        return action_logprobs, state_values, dist_entropy
