# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_features, layer1_size):
    #def __init__(self, layer1_size=64, layer2_size=84):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, layer1_size)
        #self.fc2 = nn.Linear(layer1_size, layer2_size)
        #self.fc3 = nn.Linear(layer2_size, 10)
        self.fc2 = nn.Linear(layer1_size,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.fc2(x)
        return x # check if you have to specify x.to(device)