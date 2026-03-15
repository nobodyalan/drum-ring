import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # return x[:, :, : -self.chomp_size].contiguous()
        return x[:, :, self.chomp_size :].contiguous()

class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
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


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNLSTMStreaming(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.lstm = nn.LSTM(num_channels[-1], hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

        # 缓存TCN输入序列
        self.tcn_cache = []
        self.hx = None  # LSTM隐藏状态

        self.receptive_field = 1
        for i in range(len(num_channels)):
            self.receptive_field += (kernel_size - 1) * (2 ** i)

    def forward(self, x):
        """
        x: (b, seq_len, input_size)
        """
        b, seq_len, input_size = x.shape

        # 更新缓存
        for t in range(seq_len):
            self.tcn_cache.append(x[:, t, :])
        # 保持缓存长度 <= 感受野
        if len(self.tcn_cache) > self.receptive_field:
            self.tcn_cache = self.tcn_cache[-self.receptive_field:]
        else:
            return torch.zeros((1,1,2))

        # 构造TCN输入
        tcn_input = torch.stack(self.tcn_cache, dim=1)  # (b, cached_len, input_size)
        tcn_out = self.tcn(tcn_input.transpose(1, 2)).transpose(1, 2)  # (b, cached_len, channels)

        # LSTM只处理中间时间步（也可以改成处理整个 seq_len）
        lstm_input = tcn_out[:, self.receptive_field // 2:self.receptive_field // 2 + 1, :]  # (b, 1, channels)
        lstm_out, self.hx = self.lstm(lstm_input, self.hx)
        out = F.relu(self.fc1(lstm_out))
        out = self.fc2(out)
        return out