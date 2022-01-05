import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN2(nn.Module):
	def __init__(self, state_size, action_size):
		super(DQN, self).__init__()
		self.first_two_layers = nn.Sequential(
			nn.Linear(state_size, 256),
			nn.ELU(),
			nn.Linear(256, 256),
			nn.ELU()
		)
		self.lstm = nn.LSTM(256, 256, 1, batch_first=True)
		self.last_linear = nn.Linear(256, action_size)

# Data Flow Protocol:
# 1. network input shape: (batch_size, seq_length, num_features)
# 2. LSTM output shape: (batch_size, seq_length, hidden_size)
# 3. Linear input shape:  (batch_size * seq_length, hidden_size)
# 4. Linear output: (batch_size * seq_length, out_size)

	def forward(self, input):
		# rint(input.size())
		x = self.first_two_layers(input)
		
		lstm_out, hs = self.lstm(x)

		batch_size, seq_len, mid_dim = lstm_out.shape
		linear_in = lstm_out.contiguous().view(seq_len * batch_size, mid_dim)
		# linear_in = lstm_out.contiguous().view(-1, lstm_out.size(2))

		# linear_in = lstm_out.reshape(-1, hidden_size) 
		return self.last_linear(linear_in)


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.main = nn.Sequential(
			nn.Linear(state_size, 64),
			nn.LeakyReLU(0.01, inplace=True),
			nn.Linear(64, 32),
			nn.LeakyReLU(0.01, inplace=True),
			nn.Linear(32, 8),
			nn.LeakyReLU(0.01, inplace=True),
			nn.Linear(8, action_size),
		)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.main(x)