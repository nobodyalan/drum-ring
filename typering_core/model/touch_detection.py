import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple, List


class TouchDetectionModel(nn.Module):
    ''' A simple MLP model.
    '''
    
    
    def __init__(self, n_in:int, n_out:int):
        super(TouchDetectionModel, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.mlp = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_out))
        
    
    def forward(self, x:Tensor):
        x = x.flatten(1)
        x = self.mlp(x)
        return x
    
    
class TouchDetectionModel2(nn.Module):
    ''' A simple CNN + MLP model.
    '''
    
    
    def __init__(self, n_in:int, n_out:int):
        super(TouchDetectionModel2, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2))
        self.mlp = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_out))
        
    
    def forward(self, x:Tensor):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x   
    
    
class TouchDetectionModel3(nn.Module):
    ''' A CNN + MLP model, modified from LeNet-5.
    '''
    
    
    def __init__(self, n_in:int, n_out:int, dropout:float=0.0):
        super(TouchDetectionModel3, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5))
        self.mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(192, 128), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_out))
        
    
    def forward(self, x:Tensor):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x    


class TouchDetectionModel4(nn.Module):
    ''' A deep CNN + MLP model, modified from VGGNet.
    '''   

    def __init__(self, n_in:int, n_out:int, dropout:float=0.0):
        super(TouchDetectionModel4, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=0))
        self.mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(384, 128), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_out))
        
    
    def forward(self, x:Tensor):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x
    

class Conv1dResBlock(nn.Module):
    ''' A residual block for 1D convolutional layers.
    '''
    
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int=3):
        super(Conv1dResBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding), nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    
    def forward(self, x:Tensor):
        res = x
        x = self.conv(x)
        x = x + self.shortcut(res)
        return x
    

class TouchDetectionModel5(nn.Module):
    ''' A deep CNN + MLP model, TD4 + residual connection.
    '''   

    def __init__(self, n_in:int, n_out:int, dropout:float=0.0):
        super(TouchDetectionModel5, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.conv = nn.Sequential(
            Conv1dResBlock(in_channels=6, out_channels=32, kernel_size=3), nn.ReLU(),
            Conv1dResBlock(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            Conv1dResBlock(in_channels=32, out_channels=64, kernel_size=3), nn.ReLU(),
            Conv1dResBlock(in_channels=64, out_channels=64, kernel_size=3), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            Conv1dResBlock(in_channels=64, out_channels=128, kernel_size=3), nn.ReLU(),
            Conv1dResBlock(in_channels=128, out_channels=128, kernel_size=3), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=0))
        self.mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(384, 128), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_out))
        
    
    def forward(self, x:Tensor):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x
    

class InceptionBlock(nn.Module):
    ''' An inception-v1 block for 1D convolution.
    '''
    
    def __init__(self, n_in:int, n_conv1:int, n_conv3:Tuple[int], n_conv5:Tuple[int], n_pool:int, alpha:float=0.01):
        super(InceptionBlock, self).__init__()
        self.alpha = alpha
        self.conv1 = nn.Conv1d(n_in, n_conv1, kernel_size=1)
        self.conv3 = nn.Sequential(nn.Conv1d(n_in, n_conv3[0], kernel_size=1), nn.LeakyReLU(alpha),
            nn.Conv1d(n_conv3[0], n_conv3[1], kernel_size=3, padding=1))
        self.conv5 = nn.Sequential(nn.Conv1d(n_in, n_conv5[0], kernel_size=1), nn.LeakyReLU(alpha),
            nn.Conv1d(n_conv5[0], n_conv5[1], kernel_size=5, padding=2))
        self.pool = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(n_in, n_pool, kernel_size=1))
        
    
    def forward(self, x:Tensor):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        xp = self.pool(x)
        x = torch.cat([x1, x3, x5, xp], dim=1)
        return x
    

class InceptionResBlock(nn.Module):
    ''' An inception-v1 block for 1D convolution with residual connection.
    '''
    
    def __init__(self, n_in:int, n_conv1:int, n_conv3:Tuple[int], n_conv5:Tuple[int], n_pool:int):
        super(InceptionResBlock, self).__init__()
        self.n_in = n_in
        self.n_out = n_conv1 + n_conv3[1] + n_conv5[1] + n_pool
        self.conv1 = nn.Conv1d(n_in, n_conv1, kernel_size=1)
        self.conv3 = nn.Sequential(nn.Conv1d(n_in, n_conv3[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(n_conv3[0], n_conv3[1], kernel_size=3, padding=1))
        self.conv5 = nn.Sequential(nn.Conv1d(n_in, n_conv5[0], kernel_size=1), nn.ReLU(),
            nn.Conv1d(n_conv5[0], n_conv5[1], kernel_size=5, padding=2))
        self.pool = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(n_in, n_pool, kernel_size=1))
        self.shortcut = nn.Identity()
        if self.n_in != self.n_out:
            self.shortcut = nn.Conv1d(self.n_in, self.n_out, kernel_size=1)
        
    
    def forward(self, x:Tensor):
        res = x
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        xp = self.pool(x)
        x = torch.cat([x1, x3, x5, xp], dim=1)
        x = x + self.shortcut(res)
        return x
    

class TouchDetectionModel6(nn.Module):
    ''' A CNN + MLP model with inception blocks.
    '''   

    def __init__(self, n_in:int, n_out:int, dropout:float=0.0, alpha:float=0.01):
        super(TouchDetectionModel6, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.alpha = alpha
        self.conv = nn.Sequential(
            InceptionBlock(n_in=6, n_conv1=16, n_conv3=(8,16), n_conv5=(8,16), n_pool=16, alpha=alpha),
            nn.BatchNorm1d(64),
            InceptionBlock(n_in=64, n_conv1=16, n_conv3=(8,16), n_conv5=(8,16), n_pool=16, alpha=alpha),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            InceptionBlock(n_in=64, n_conv1=32, n_conv3=(16,32), n_conv5=(16,32), n_pool=32, alpha=alpha),
            nn.BatchNorm1d(128),
            InceptionBlock(n_in=128, n_conv1=32, n_conv3=(16,32), n_conv5=(16,32), n_pool=32, alpha=alpha),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
            InceptionBlock(n_in=128, n_conv1=64, n_conv3=(32,64), n_conv5=(32,64), n_pool=64, alpha=alpha),
            nn.BatchNorm1d(256),
            InceptionBlock(n_in=256, n_conv1=64, n_conv3=(32,64), n_conv5=(32,64), n_pool=64, alpha=alpha),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=0))
        self.mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(384, 128), nn.LeakyReLU(alpha),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64), nn.LeakyReLU(alpha),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32), nn.LeakyReLU(alpha),
            nn.Linear(32, n_out))
        
    
    def forward(self, x:Tensor):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x
    

    def cnn_l2_loss(self):
        loss = 0.0
        for p in self.conv.parameters():
            loss += torch.sum(p**2)
        return loss
    

    def mlp_l2_loss(self):
        loss = 0.0
        for p in self.mlp.parameters():
            loss += torch.sum(p**2)
        return loss
        

class TouchDetectionModel7(nn.Module):
    ''' Reimplementation of TapType model.
    '''

    def __init__(self, n_in:int, n_out:int, dropout:float=0.0, alpha:float=0.01):
        super(TouchDetectionModel7, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.alpha = alpha
        self.conv = nn.Sequential(
            nn.Conv1d(6, 16, 3, padding=1), nn.BatchNorm1d(16), nn.LeakyReLU(alpha),
            nn.Conv1d(16, 16, 3, padding=1), nn.BatchNorm1d(16), nn.LeakyReLU(alpha),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.LeakyReLU(alpha),
            nn.Conv1d(32, 32, 3, padding=1), nn.BatchNorm1d(32), nn.LeakyReLU(alpha),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(alpha),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(alpha),
            nn.AvgPool1d(2))
        self.mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(320, 80), nn.LeakyReLU(alpha),
            nn.Dropout(self.dropout),
            nn.Linear(80, 20), nn.LeakyReLU(alpha),
            nn.Dropout(self.dropout),
            nn.Linear(20, n_out))
        

    def forward(self, x:Tensor):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x
    

class LayerNorm1d(nn.Module):
    r""" LayerNorm that supports two data formats: channel_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channel_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    
    def __init__(self, normalized_shape, eps=1e-6, data_format='channel_first'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channel_last', 'channel_first']:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    
    def forward(self, x):
        if self.data_format == 'channel_last':  # [B, T, C]
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == 'channel_first':  # [B, C, T]
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x
    
    
class TouchDetectionModel8(nn.Module):
    ''' A CNN + MLP model with inception blocks.
    '''   

    def __init__(self, n_in:int, n_out:int, dropout:float=0.0, alpha:float=0.01):
        super(TouchDetectionModel8, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dropout = dropout
        self.alpha = alpha
        self.conv = nn.Sequential(
            InceptionBlock(n_in=n_in, n_conv1=16, n_conv3=(16,16), n_conv5=(16,16), n_pool=16, alpha=alpha),
            LayerNorm1d(64, data_format='channel_first'),
            InceptionBlock(n_in=64, n_conv1=16, n_conv3=(16,16), n_conv5=(16,16), n_pool=16, alpha=alpha),
            LayerNorm1d(64, data_format='channel_first'),
            InceptionBlock(n_in=64, n_conv1=32, n_conv3=(32,32), n_conv5=(32,32), n_pool=32, alpha=alpha),
            LayerNorm1d(128, data_format='channel_first'),
            InceptionBlock(n_in=128, n_conv1=32, n_conv3=(32,32), n_conv5=(32,32), n_pool=32, alpha=alpha),
            LayerNorm1d(128, data_format='channel_first'),
            nn.MaxPool1d(kernel_size=2),
            InceptionBlock(n_in=128, n_conv1=64, n_conv3=(64,64), n_conv5=(64,64), n_pool=64, alpha=alpha),
            LayerNorm1d(256, data_format='channel_first'),
            InceptionBlock(n_in=256, n_conv1=64, n_conv3=(64,64), n_conv5=(64,64), n_pool=64, alpha=alpha),
            LayerNorm1d(256, data_format='channel_first'),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=0))
        self.mlp = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(384, 128), nn.LeakyReLU(alpha),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64), nn.LeakyReLU(alpha),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32), nn.LeakyReLU(alpha),
            nn.Linear(32, n_out))
        
    
    def forward(self, x:Tensor):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.mlp(x)
        return x
    

    def cnn_l2_loss(self):
        loss = 0.0
        for p in self.conv.parameters():
            loss += torch.sum(p**2)
        return loss
    

    def mlp_l2_loss(self):
        loss = 0.0
        for p in self.mlp.parameters():
            loss += torch.sum(p**2)
        return loss


if __name__ == '__main__':
    model = TouchDetectionModel8(6, 5)
    x = torch.randn(2, 20, 6)
    out = model(x)
    print(out.shape)