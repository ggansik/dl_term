
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np



class HighwayNet(nn.Module):
    """
    h * t + x * (1. - t)
    """
    def __init__(self):
        super(HighwayNet, self).__init__()
        
        self.H = nn.Linear(128, 128)
        self.T = nn.Linear(128, 128)
        
    def forward(self, x):
        h = F.relu(self.H(x))
        t = torch.sigmoid(self.T(x))
        
        output = h * t + x * (1. - t)
        return output



class EncoderCBHG(nn.Module):
    """
    Conv1D bank - Max Pooling - Conv1D projection - Conv1D Layer
    """
    def __init__(self, K=16):
        super(EncoderCBHG, self).__init__()
        #conv1d: (batch_size, Channel, Length)
        #-----------------Conv1Dbank-------------------#
        self.conv1dBank = nn.ModuleList(
            [nn.Conv1d(128, 128, k, stride=1, padding=k//2)
            for k in range(1, K+1)]
        )
        #-----------------Max pooling------------------#
        self.maxPool = nn.MaxPool1d(2, stride=1, padding=1)
        #---------------Conv1Dprojection---------------#
        self.conv1dProjs = nn.ModuleList(
            [nn.Conv1d(128 * K, 128, 3, stride=1, padding=1)
            ,nn.Conv1d(128, 128, 3, stride=1, padding=1)]
        )
        #-----------------Highway Net------------------#
        self.highwayNet = nn.ModuleList(
            [HighwayNet() for _ in range(4)]
        )
        #--------------Bidirectional GRU---------------#
        self.GRU = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        #-------------Batch normalization--------------#
        self.bn = nn.BatchNorm1d(128)
        
    def forward(self, x):
        """
        Args:
            x: A tensor with shape (batch_size, seq_length, channels)
        Returns:
            A tensor with shape (batch_size, seq_length, 2*hidden_size)
        """
        # modify the shape of x to (batch_size, channels, seq_length)
    
        x = x.T(1, 2) # Shape: (batch_size, channels, seq_length)
        #-----------------Conv1Dbank-------------------#
        stacked = []
        for conv1d in conv1dBank:
            stacked.append(self.bn(conv1d(x)))
        stacked = torch.cat(stacked, dim=1)
        #shape: 
        #-----------------Max pooling------------------#
        y = self.maxPool(stacked)
        #---------------Conv1Dprojection---------------#
        y = self.bn(self.relu(self.conv1dProjs[0](y)))
        y = self.bn(self.conv1dProjs[1](y))
        #-------------residual connection--------------#
        y = y + x
        #----------------Highway Net-------------------#
        for layer in self.highwayNet:
            y = self.relu(layer(y))
        #--------------Bidirectional GRU---------------#
        y, _ = self.GRU(y)
        
        return y



class Prenet(nn.Module):
    """
        FC(Dense) - ReLU - Dropout - FC - ReLU - Dropout
    """
    def __init__(self):
        super(Prenet, self).__init__()
        self.layer = nn.ModuleList(
            [nn.Linear(256, 256)
            ,nn.Linear(256, 128)]
        )
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        for layer in self.layer:
            x = self.dropout(F.relu(layer(x)))
        
        return x


class Encoder(nn.Module):
    """
        prenet - CBHG
    """
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.prenet = Prenet()
        self.cbhg = EncoderCBHG()
    
    def forward(self, x):
        x = self.cbhg(self.prenet(x))
        
        return x
        


