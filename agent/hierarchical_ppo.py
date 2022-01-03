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

        print("state size in bid network {}".format(state_size))

        self.layer1=nn.Linear(state_size, 128)
        self.tanh = nn.Tanh()

        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_size)
        )

        self.sm = nn.Softmax(dim=-1) # check if softmax because not in paper

    def forward(self, x):
        print("x : {}".format(x))
        #  shape [batch_size, nb_features], NOT [batch_size, nb_features, 1]
        x_viewed = x.view(x.size(0), -1)
        print("X viewed in bid net input : {}".format(x_viewed.shape))
        # with torch.no_grad():
        #     x_viewed.clamp_(min=1e-2)
        print("X viewed clamped in bid net input : {}".format(x_viewed.shape))
        layer1 = self.layer1(x_viewed)
        print("layer1 {}".format(layer1))
        tanh = self.tanh(layer1)
        print("tanh {}".format(tanh))
        layers = self.layers(tanh)
        print("layers {}".format(layers))
        softmax = self.sm(layers)
        print("softmaxed {}".format(layers))
        return softmax


class Actor(nn.Module):
    def __init__(self, state_size, action_std_init=0.6):
        super(Actor, self).__init__()
        self.order_net = OrderNetwork(state_size, 1)
        self.bid_net = BidNetwork(state_size+1, 3)

        self.action_var = torch.full(
            (1,), action_std_init * action_std_init).to(device)

    def act(self, state):
        print("state in actor {}".format(state.shape))
        amount_order_net = self.order_net(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(amount_order_net, cov_mat)
        amount = torch.clip(dist.sample(), min=0, max=1)
        print("Sampled amount {}".format(amount))
        amount_logprob = dist.log_prob(amount)  # maybe to use later

        # Bid network takes as input [observed state, sampled amount]
        state_with_amount = torch.cat((state.view(state.size(0),-1), amount), dim=1)

        print("state with amount in actor {}".format(state_with_amount.shape))

        print("Check finiteness stateamount {}".format(torch.all(torch.isfinite(state_with_amount))))


        print("VALID : {}".format(torch.all(torch.isfinite(state_with_amount[0])).item()))

        out_bid_net = self.bid_net(state_with_amount)

        print("Out bid net {}".format(out_bid_net))

        return amount, out_bid_net


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, action_std_init, window_size):
        super(ActorCritic, self).__init__()

        self.action_size = action_size

        self.actor = Actor(state_size * window_size, action_std_init)

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size * window_size + 1, 128),
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
        print("state -> {}".format(state.shape))
        print(state)

        print("Check finiteness state {}".format(torch.all(torch.isfinite(state))))
        action_actor = self.actor.act(state)
        print("action actor -> {}".format(action_actor))
        amount = action_actor[0]
        action_type_probs = action_actor[1].squeeze()
        dist = Categorical(action_type_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        print("state viewed -> {}".format(state.view(state.size(0), -1).shape))
        print("amount -> {}".format(amount.shape))
        print("state w/ amount -> {}".format(torch.cat((state.view(state.size(0), -1), amount), dim=1).shape))

        state_with_amount = torch.cat((state.view(state.size(0), -1), amount), dim=1)
        state_values = self.critic(state_with_amount)

        return action_logprobs, state_values, dist_entropy
