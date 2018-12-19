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

class DecoderCBHG(nn.Module):
    """
    Conv1D bank - Max Pooling - Conv1D projection - Conv1D Layer
    """
    def __init__(self, K=8):
        super(DecoderCBHG, self).__init__()
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
            [nn.Conv1d(128 * K, 256, 3, stride=1, padding=1)
            ,nn.Conv1d(256, 80, 3, stride=1, padding=1)]
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

             
class AttentionWrapper(nn.Module):
    def __init__(self, rnn, use_attention):
        super(AttentionWrapper, self).__init__()
        self.rnn_cell = rnn
        self.attention = use_attention
        self.projection_for_decoderRNN = nn.Linear(512, 256, bias=False)
    def forward(memory, decoder_input, cell_hidden):
        """
        memory = (batch_size, encoder_T, dim)
        decoder_input = (batch_size, dim)
        cell_hidden (previous time step cell state) = (batch, dim)
        """
        #cell_input = torch.cat((decoder_input, prev_attention), -1) -- why do we have to concat?
        cell_input = decoder_input
        query = self.rnn_cell(cell_input, cell_hidden)
        #feed into attention
        attention_weights = self.attention(query, memory)
        #make context vector
        attention_weights = F.softmax(attention_weights)
        context = torch.bmm(attention_weights.view(batch, 1, -1), memory).squeeze(1)
        out = self.projection_for_decoderRNN(torch.cat([context, query],dim=-1))
        return out, query, attention_weights


class BahnadauAttention(nn.Module):
    def __init__(self):
        super(BahnadauAttention, self).__init__()
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
    def __init__(self, spect_dim, r=2):
        super(Decoder, self).__init__()
        self.spect_dim = spect_dim
        self.r = r
        self.prenet = Prenet()
        self.attention_RNN = AttentionWrapper(nn.GRUCell(input_size=256, hidden_size =256), BahdanauAttention())
        self.decoder_RNN = nn.ModuleList(
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
            target = target.view(batch_size, target.size(1) // r, -1)
            decoder_T = target.size(1)
            target = target.transpose(0,1) #for parallelization
            
        #2단계 decoderRNN 값 저장할 array
        decoderRNN_output = [memory.zero_() for _ in range(len(decoder_RNN))] 
        
        #<GO> Frame
        current_input = torch.zero([batch_size, self.r*self.spect_dim])
        t = 0
        targets = []
        attention_weights = []
        
        while (True):
            t = t + 1
            #prenet
            #(B, spect_dim * r)
            prenet_output = self.prenet(current_input)
            
            #attention
            #(B, 256)
            attention_output, cell_hidden, attention_weight = self.attention(memory, prenet_output, cell_hidden)
            
            #decoder
            #(B, spect_dim * r)
            for idx in range(2):
                decoderRNN_output[idx] = self.decoder_RNN[idx](attention_output, decoder_output[idx])
                decoderRNN_output[idx] += attention_output
                attention_output = decoder_output[idx]
            
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
        
        attention_weights = torch.stack(attention_weights).transpose(0,1)
        outputs = torch.stack(outputs).transpose(0,1).contiguous()
        return outputs, attention_weights
