import torch
import torch.nn as nn

class MetricNet(nn.Module):
    def __init__(self, input_size, num_classes=(False, 10), p=1):
        super().__init__()
        
        hidden_size1 = int(input_size/2)
        hidden_size2 = int(input_size/4)
        hidden_size3 = int(input_size/8)
        hidden_size4 = int(input_size/16)
        hidden_size5 = int(input_size/32)
        
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, hidden_size1)
        self.fc3 = nn.Linear(hidden_size1, hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, hidden_size3)
        self.fc5 = nn.Linear(hidden_size3, hidden_size4)
        self.fc6 = nn.Linear(hidden_size4, hidden_size5)
        if num_classes[0]:

            self.fc7 = nn.Linear(hidden_size5, num_classes[1])
        else:

            self.fc7 = nn.Linear(hidden_size5, p)
        
        
        self.relu = nn.ReLU()

        # TESTING ONLY
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.fc7(x)
        return x