import torch
from torch import nn
from torch.nn import init

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, seq_length = 100):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * seq_length, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.init_weights()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def init_weights(self):    
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                elif 'fc' in name:
                    init.kaiming_uniform_(param, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out.flatten(start_dim=1))
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        # print("After softmax",out)
        return out
