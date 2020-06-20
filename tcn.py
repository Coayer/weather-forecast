import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(ResidualBlock, self).__init__()
        conv1 = conv2 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                       stride=stride, padding=padding, dilation=dilation))

        self.net = nn.Sequential(conv1, nn.ZeroPad2d((0, -padding, 0, 0)), nn.ReLU(), nn.Dropout(dropout),
                                 conv2, nn.ZeroPad2d((0, -padding, 0, 0)), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x) + x


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1)* dilation_size, dropout=dropout)) # padding = ... * dilation_size

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
