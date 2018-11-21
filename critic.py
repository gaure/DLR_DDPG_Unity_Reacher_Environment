import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 fc1_units=64,
                 fc2_units=128,
                 fc_aggregate_units=32):
        """

        :state_size: (int) Dimension of each state
        :action_size: (int) Dimension of each action
        :fc1_units: (int) number of units of first hidden layer
        :fc2_units: (int) number of units of first hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        # critic state hidden layers
        self.state_fc1 = nn.Linear(self.state_size, fc1_units)
        self.state_fc2 = nn.Linear(fc1_units, fc2_units)

        # critic action hidden layers
        self.action_fc1 = nn.Linear(self.action_size, fc1_units)
        self.action_fc2 = nn.Linear(fc1_units, fc2_units)

        # critic aggregate layers
        self.fc_aggregate = nn.Linear(fc2_units, fc_aggregate_units)
        self.fc_output = nn.Linear(fc_aggregate_units, 1)

    def forward(self, state, action):
        # state path
        s = F.relu(self.state_fc1(state))
        s = F.relu(self.state_fc2(s))

        # action path
        a = F.relu(self.action_fc1(action))
        a = F.relu(self.action_fc2(a))

        # merge
        #x = s + a
        x = torch.cat([a, s],0)
        x = F.relu(self.fc_aggregate(x))
        q_values = self.fc_output(x)

        return q_values
