# Cell
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .common import Chomp1d
from .common import CausalConv1d

# Cell
# https://github.com/locuslab/TCN
class _TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(_TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Cell
class _TemporalBlock2(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(_TemporalBlock2, self).__init__()
        self.causalconv1 = CausalConv1d(in_channels=n_inputs, out_channels=n_outputs,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, activation='ReLU', with_weight_norm=True)

        self.causalconv2 = CausalConv1d(in_channels=n_outputs, out_channels=n_outputs,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        dilation=dilation, activation='ReLU', with_weight_norm=True)

        self.net = nn.Sequential(self.causalconv1, nn.Dropout(dropout),
                                 self.causalconv2, nn.Dropout(dropout))

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.causalconv1.conv.weight.data.normal_(0, 0.01)
        self.causalconv2.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Cell
class _TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(_TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            #layers += [_TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
            #                        padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            layers += [_TemporalBlock2(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)