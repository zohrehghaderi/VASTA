import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_size, hidden_sizes, out_size, dropout_p=0.5, have_last_bn=False,
                 pretrained_model_path=''):
        super(MLP, self).__init__()

        self.in_drop = nn.Dropout(dropout_p)

        self.fc1 = nn.Linear(in_size, hidden_sizes[0])
        self.fc1_drop = nn.Dropout(dropout_p)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2_drop = nn.Dropout(dropout_p)

        self.fc3 = nn.Linear(hidden_sizes[1], out_size)

        self.have_last_bn = have_last_bn
        if have_last_bn:
            self.bn = nn.BatchNorm1d(out_size)
        self.maxpool=nn.MaxPool1d(16)
        self.__init_layers()

    def __init_layers(self):
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)


    def forward(self, x):
        h = self.in_drop(x)
        h = self.fc1_drop(torch.relu(self.fc1(h)))
        h = self.fc2_drop(torch.relu(self.fc2(h)))
        h = self.fc3(h)
        h= h.max(dim=1).values
        return h




