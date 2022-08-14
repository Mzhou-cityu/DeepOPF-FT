import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_units, khidden):
        super(Net, self).__init__()
        self.num_layer = khidden.shape[0]
        self.fc1 = nn.Linear(input_channels, khidden[0] * hidden_units)
        if self.num_layer >= 2:
            self.fc2 = nn.Linear(khidden[0] * hidden_units, khidden[1] * hidden_units)
        if self.num_layer >= 3:
            self.fc3 = nn.Linear(khidden[1] * hidden_units, khidden[2] * hidden_units)
        if self.num_layer >= 4:
            self.fc4 = nn.Linear(khidden[2] * hidden_units, khidden[3] * hidden_units)
        self.fcbfend = nn.Linear(khidden[khidden.shape[0] - 1] * hidden_units, output_channels)
        self.fcend = nn.Linear(output_channels, output_channels)

    def forward(self, x):
        x_h = F.relu(self.fc1(x))
        if self.num_layer >= 2:
            x_h= F.relu(self.fc2(x_h))
        if self.num_layer >= 3:
            x_h = F.relu(self.fc3(x_h))
        if self.num_layer >= 4:
            x_h = F.relu(self.fc4(x_h))
        # fixed final two layers
        x_h = F.relu(self.fcbfend(x_h))
        x_Pred = F.sigmoid(self.fcend(x_h))
        return x_Pred
