#Total model - Tacotron whole model
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


class PostProcessing(nn.Module):
    """
    make post processing data
    input : (B, decoder.T, spect_dim)
    """
    def __init__(self, spect_dim):
        super(PostProcessing, self).__init__()
        self.postcbhg = DecoderCBHG(80, K=8)
        self.linear = nn.Linear(256, 1025)
    def forward(self, batch_size, data):
        """
            make data shape (B, -1, 80)
        """
        output = self.postcbhg(data)
        print("after postcbhg", end= " ")
        print(output.size())
        output = self.linear(output)
        print("after postlinear", end=" ")
        print(output.size())
        
        return output


class Tacotron(nn.Module):
    def __init__(self, vocab_num, input_dim=256, spect_dim=80):
        super(Tacotron, self).__init__()
        self.input_dim = input_dim
        self.spect_dim = spect_dim
        self.embedding = nn.Embedding(vocab_num, input_dim) #embedding dimension
        self.embedding.weight.data.normal_(0,0.3)
        self.Encoder = Encoder()
        self.Decoder = Decoder(spect_dim, r=5) #write input_dimension
        self.PostProcessing = PostProcessing(spect_dim)
    def forward(self, inputs, spect_targets=None, r= 5):
        """
        make total model!
        input : (B, encoder.T, in_dim)
        
        """
        print("before embedding", end=" ")
        print(inputs.size())
        batch_size = inputs.size(0)
        memory = self.embedding(inputs)
        print("after embedding", end=" ")
        print(memory.size())
        #encoding
        #(B, encoder.T, input_dim)
        memory = self.Encoder(memory)
        print("after encoding", end=" ")
        print(memory.size())
        #decoding
        #(B, encoder.T, mel_dim * r)
        decoder_output, attention_weights = self.Decoder(memory, spect_targets)
        
        #postprocessing
        #(B, decoder.T, 1025)
        decoder_output = decoder_output.view(batch_size, -1, self.spect_dim)
        wav_output = self.PostProcessing(batch_size, decoder_output)
        
        return decoder_output, wav_output, attention_weights