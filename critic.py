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
                 fc3_units=64):
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
        self.sln1 = nn.LayerNorm(fc1_units)
        self.state_fc2 = nn.Linear(fc1_units, fc2_units)
        self.sln2 = nn.LayerNorm(fc2_units)

        # critic action hidden layers
        self.action_fc1 = nn.Linear(self.action_size, fc1_units)
        self.aln1 = nn.LayerNorm(fc1_units)
        self.action_fc2 = nn.Linear(fc1_units, fc2_units)
        self.aln2 = nn.LayerNorm(fc2_units)

        # critic aggregate layers
        self.aggregate_fc1 = nn.Linear(fc2_units, fc3_units)
        self.aggregate_ln1 = nn.LayerNorm(fc3_units)

        # critic output
        self.fc_output = nn.Linear(fc3_units, 1)

    def forward(self, state, action):
        # state path
        s = F.relu(self.sln1(self.state_fc1(state)))
        s = F.relu(self.sln2(self.state_fc2(s)))

        # action path
        a = F.relu(self.aln1(self.action_fc1(action)))
        a = F.relu(self.aln2(self.action_fc2(a)))

        # merge by adding both path and producting tensor same size
        # equivalent to keras.Add()
        x = s + a

        # Pass the above result through a relu function and produce
        # a one output node with no activation.
        x = F.relu(self.aggregate_ln1(self.aggregate_fc1(x)))
        q_values = self.fc_output(x)

        return q_values
