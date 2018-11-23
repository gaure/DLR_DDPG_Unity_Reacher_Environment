import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):

    def __init__(self,
                 state_size,
                 action_size,
                 seed,
                 fc1_units=32,
                 fc2_units=64):
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
        self.sdo1 = nn.Dropout(0.2)
        self.state_fc2 = nn.Linear(fc1_units, fc2_units)
        self.sln2 = nn.LayerNorm(fc2_units)
        self.sdo2 = nn.Dropout(0.2)

        # critic action hidden layers
        self.action_fc1 = nn.Linear(self.action_size, fc1_units)
        self.aln1 = nn.LayerNorm(fc1_units)
        self.ado1 = nn.Dropout(0.2)
        self.action_fc2 = nn.Linear(fc1_units, fc2_units)
        self.aln2 = nn.LayerNorm(fc2_units)
        self.ado2 = nn.Dropout(0.2)

        # critic aggregate layers
        self.fc_output = nn.Linear(fc2_units, 1)
        self.odo = nn.Dropout(0.2)

    def forward(self, state, action):
        # state path
        s = F.relu(self.sdo1(self.sln1(self.state_fc1(state))))
        s = F.relu(self.sdo2(self.sln2(self.state_fc2(s))))

        # action path
        a = F.relu(self.ado1(self.aln1(self.action_fc1(action))))
        a = F.relu(self.ado2(self.aln2(self.action_fc2(a))))

        # merge by adding both path and producting tensor same size
        # equivalent to keras.Add()
        x = s + a

        # Pass the above result through a relu function and produce
        # a one output node with no activation.
        q_values = self.odo(self.fc_output(F.relu(x)))

        return q_values
