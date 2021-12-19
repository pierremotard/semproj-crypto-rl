class RolloutBuffer(object):
    def __init__(self):
        self.action_types= []
        self.amounts = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear(self):
        del self.action_types[:]
        del self.amounts[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

