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



class DecoderCBHG(nn.Module):
    """
    Conv1D bank - Max Pooling - Conv1D projection - Conv1D Layer
    """
    def __init__(self, in_dim, K=8):
        super(DecoderCBHG, self).__init__()
        #conv1d: (batch_size, Channel, Length)
        #-----------------Conv1Dbank-------------------#
        self.conv1dBank = nn.ModuleList(
            [nn.Conv1d(in_dim, 128, k, stride=1, padding=k//2) for k in range(1, K+1)]
        )
        #-----------------Max pooling------------------#
        self.maxPool = nn.MaxPool1d(2, stride=1, padding=1)
        #---------------Conv1Dprojection---------------#
        self.conv1dProjs = nn.ModuleList(
            [nn.Conv1d(128 * K, 256, 3, stride=1, padding=1)
            ,nn.Conv1d(256, 80, 3, stride=1, padding=1)]
        )
        #-----------------Highway Net------------------#
        self.highwayNet = nn.ModuleList(
            [HighwayNet(80) for _ in range(4)]
        )
        #--------------Bidirectional GRU---------------#
        self.GRU = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        #-------------Batch normalization--------------#
        self.bn = nn.BatchNorm1d(128)
        self.bn_proj1 = nn.BatchNorm1d(256)
        self.bn_proj2 = nn.BatchNorm1d(80)
        #------------Transformation for Highwaynet-----#
        self.high_linear = nn.Linear(in_dim, 128)
        
    def forward(self, x):
    
        x = x.transpose(1, 2) # Shape to: (batch_size, channels, seq_length)
        #-----------------Conv1Dbank-------------------#
        temp = x.size(-1)
        stacked = []
        for conv1d in self.conv1dBank:
            stacked.append(self.bn(conv1d(x)[:, :, :temp]))
        
        stacked = torch.cat(stacked, dim=1)
        #shape: 
        #-----------------Max pooling------------------#
        y = self.maxPool(stacked)[:, :, :temp]
        #---------------Conv1Dprojection---------------#
        y = self.bn_proj1(F.relu(self.conv1dProjs[0](y)))
        y = self.bn_proj2(self.conv1dProjs[1](y))
        #-------------residual connection--------------#
        y = y + x
        #----------------Highway Net-------------------#
        y = y.transpose(1, 2)
        y = self.high_linear(y)
        for layer in self.highwayNet:
            y = F.relu(layer(y))
        #--------------Bidirectional GRU---------------#
        y, _ = self.GRU(y)
        
        return y

             
class AttentionWrapper(nn.Module):
    def __init__(self, rnn, use_attention):
        super(AttentionWrapper, self).__init__()
        self.rnn_cell = rnn
        self.attention = use_attention
        self.projection_for_decoderRNN = nn.Linear(512, 256, bias=False)
    def forward(self, memory, decoder_input, cell_hidden):
        """
        memory = (batch_size, encoder_T, dim)
        decoder_input = (batch_size, dim)
        cell_hidden (previous time step cell state) = (batch, dim)
        """
        batch_size = memory.size(0)
        #cell_input = torch.cat((decoder_input, prev_attention), -1) -- why do we have to concat?
        cell_input = decoder_input
        query = self.rnn_cell(cell_input, cell_hidden)
        #feed into attention
        attention_weights = self.attention(query, memory)
        #make context vector
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(batch_size
                                ,attention_weights.size(1))
        context = torch.bmm(attention_weights.view(batch_size, 1, -1), memory).squeeze(1)
        out = self.projection_for_decoderRNN(torch.cat([context, query],dim=-1))
        return out, query, attention_weights


class BahdanauAttention(nn.Module):
    def __init__(self):
        super(BahdanauAttention, self).__init__()
        self.v = nn.Linear(256,1,bias=False)
        self.query_layer = nn.Linear(256,256,bias=False)
        self.tanh = nn.Tanh()
    def forward(self, query, memory):
        """
        query : (batch, 1 ,dim)
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)
        attention_weight = self.v(self.tanh(self.query_layer(query) + memory))
        return attention_weight


class Decoder(nn.Module):
    def __init__(self, spect_dim, r=5):
        super(Decoder, self).__init__()
        self.spect_dim = spect_dim
        self.r = r
        self.prenet = Prenet(r*spect_dim)
        self.attention_RNN = AttentionWrapper(nn.GRUCell(input_size=128, hidden_size =256), BahdanauAttention())
        self.decoderRNN = nn.ModuleList(
                            [nn.GRUCell(input_size=256,hidden_size=256) for _ in range(2)])
        self.spectro_layer = nn.Linear(256,spect_dim*r,bias=False)
        self.epsilon = 0.2
        self.maximum_step = 1000
        return
    
    def forward(self, memory, target=None):
        """
        if training time, input is given, else input is decoder outputs
        input : 
            memory (encoder_output) = (batch_size, encoder_T, char_dim)
            decoder_input = (batch_size, decoder_T, dim)
        output:
            
        """
        batch_size = memory.size(0)
        test = target is None
        decoder_T = 0
        
        #train data를 r 단위로 묶어준 후 T의 크기를 바꾸어준다.
        if not test:
            target = target.view(batch_size, target.size(1) // self.r, -1)
            decoder_T = target.size(1)
            target = target.transpose(0,1) #for parallelization
            
        #2단계 decoderRNN 값 저장할 array
        decoderRNN_output = [torch.zeros([batch_size, 256]) for _ in range(len(self.decoderRNN))] 
        cell_hidden = torch.zeros([batch_size, 256])
        
        #<GO> Frame
        #print(self.r * self.spect_dim)
        current_input = torch.zeros([batch_size, self.r*self.spect_dim])
        t = 0
        targets = []
        attention_weights = []
        
        while (True):
            t = t + 1
            
            #prenet
            #(B, spect_dim * r)
            #print(current_input.size())
            prenet_output = self.prenet(current_input)
            #attention
            #(B, 256)
            attention_output, cell_hidden, attention_weight = self.attention_RNN(memory, prenet_output, cell_hidden)
    
            #decoder
            #(B, spect_dim * r)
            for idx in range(2):
                decoderRNN_output[idx] = self.decoderRNN[idx](attention_output, decoderRNN_output[idx])
                decoderRNN_output[idx] += attention_output
                attention_output = decoderRNN_output[idx]
            
            #projection
            targetchar =self.spectro_layer(attention_output)
            targets += [targetchar]
            attention_weights += [attention_weight]
            
            #check if this target is the end
            if test:
                if t > 1 and (targetchar<=self.epsilon).all(): break
                if t > self.maximum_step: 
                    print("ERROR : Not converge")
                    break
            else:
                if t >= decoder_T:
                    break
                    
            #change current input
            if test:
                current_input = targets[-1]
            else:
                current_input = target[t-1]
                
        print(t)
        attention_weights = torch.stack(attention_weights).transpose(0,1)
        
        targets = torch.stack(targets).transpose(0,1).contiguous()
        return targets, attention_weights
