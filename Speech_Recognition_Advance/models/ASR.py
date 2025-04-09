import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Permute(torch.nn.Module):
    '''
    Used to transpose/permute the dimensions of an MFCC tensor.
    '''
    def forward(self, x):
        return x.transpose(1, 2)


class pBLSTM(torch.nn.Module):

    '''
    Pyramidal BiLSTM

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, pass 1 layer at a time.
    '''

    def __init__(self, input_size, hidden_size):
        super(pBLSTM, self).__init__()

        # input size = 2 * input_size since concatenating two consecutive frames
        self.blstm = nn.LSTM(input_size=2 * input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x_packed): # x_packed is a PackedSequence

        # Pad Packed Sequence
        x, length = pad_packed_sequence(x_packed, batch_first=True)  # shape: (batch_size, t, f_dim)
        
        # Downsample the time dimension
        x, length = self.trunc_reshape(x, length)                    # shape: (batch_size, t//2, 2*f_dim)

        # Pack Padded Sequence
        x_packed = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)

        # Pass the sequence through bLSTM
        output, (hn, cn) = self.blstm(x_packed)

        return output


    def trunc_reshape(self, x, x_lens):
        batch_size, t, f_dim = x.size()
        
        # If odd number of timesteps
        if t % 2 != 0:
            x = x[:, :-1, :]
            t = t - 1
        
        new_t = t // 2

        # Reshape x
        x = x.reshape(batch_size, new_t, 2, f_dim)
        x = x.reshape(batch_size, new_t, 2 * f_dim)

        # Reduce lengths by the same downsampling factor
        new_lens = [l // 2 for l in x_lens]
        new_lens = torch.tensor(new_lens)

        return x, new_lens
    

class LSTMWrapper(torch.nn.Module):
    '''
    Used to get only output of lstm, not the hidden states.
    '''
    def __init__(self, lstm):
        super(LSTMWrapper, self).__init__()
        self.lstm = lstm

    def forward(self, x):
        output, _ = self.lstm(x)
        return output



class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, cnn_arch, input_size, encoder_hidden_size, num_pBLSTM, encoder_dropout):
        super(Encoder, self).__init__()

        # use CNNs as Embedding layer to extract features
        self.embedding = self.cnn_blk(cnn_arch)

        self.BLSTMs = LSTMWrapper(
            torch.nn.LSTM(input_size=cnn_arch[-1][0], hidden_size=encoder_hidden_size, num_layers=2, bidirectional=True)
        )

        self.pBLSTMs = torch.nn.Sequential(
            *[pBLSTM(input_size=2 * encoder_hidden_size, hidden_size=encoder_hidden_size) for _ in range(num_pBLSTM)]
        )
        
        self.layer_norm = nn.LayerNorm(2 * encoder_hidden_size)
        self.final_dropout = nn.Dropout(encoder_dropout)
        
    def forward(self, x, x_lens):
        # Where are x and x_lens coming from? The dataloader

        # shape of x: (b, t, f_dim) --> (b, f_dim, t)
        x = x.transpose(1, 2)

        x = self.embedding(x)
        x = x.transpose(1, 2)       # shape: (b, t, f_dim)

        # Pack Padded Sequence
        x_packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)

        x_packed = self.BLSTMs(x_packed)
        x_packed = self.pBLSTMs(x_packed)

        # Pad Packed Sequence
        encoder_outputs, encoder_lens = pad_packed_sequence(x_packed, batch_first=True)
        encoder_outputs = self.layer_norm(encoder_outputs)
        encoder_outputs = self.final_dropout(encoder_outputs)

        return encoder_outputs, encoder_lens
    
    def cnn_blk(self, arch): 
        '''
        Parameters
        ----------
        arch: list of list 
            [[out_channel, kernel_size, stride, padding], ...]
        '''
        layers = []

        for out_channels, kernel_size, stride, padding in arch:
            layers.append(nn.LazyConv1d(out_channels = out_channels, 
                                        kernel_size = kernel_size, 
                                        stride = stride, 
                                        padding= padding))
            
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.GELU())
        
        return nn.Sequential(*layers)



class Decoder(torch.nn.Module):

    def __init__(self, embed_size, arch, decoder_dropout, output_size=41):
        super().__init__()

        mlp_layers = [
            Permute(),
            torch.nn.BatchNorm1d(2 * embed_size),
            Permute()
        ]

        mlp_layers.append(self.linear_blk(arch))
        mlp_layers.append(nn.LazyLinear(output_size))
        mlp_layers.append(nn.Dropout(decoder_dropout))

        self.mlp = nn.Sequential(*mlp_layers)
        self.softmax = torch.nn.LogSoftmax(dim=2)

    def linear_blk(self, arch):
        blk = []

        for out, dropout in arch:
            blk.append(nn.LazyLinear(out))
            blk.append(Permute())
            blk.append(nn.LazyBatchNorm1d())
            blk.append(Permute())
            blk.append(nn.ReLU())
            if dropout:
                blk.append(nn.Dropout(dropout))
        
        return nn.Sequential(*blk)

    def forward(self, encoder_out):
        
        out = self.mlp(encoder_out)
        out = self.softmax(out)

        return out



class ASRModel(torch.nn.Module):

    def __init__(self, cnn_arch, input_size, encoder_hidden_size, num_pBLSTM, linear_arch, encoder_dropout, decoder_dropout, embed_size=192, output_size=41):
        super().__init__()

        # Initialize encoder and decoder
        self.encoder        = Encoder(cnn_arch, input_size, encoder_hidden_size, num_pBLSTM, encoder_dropout = encoder_dropout)
        self.decoder        = Decoder(embed_size, linear_arch, decoder_dropout, output_size)


    def forward(self, x, lengths_x):

        encoder_out, encoder_lens   = self.encoder(x, lengths_x)
        decoder_out                 = self.decoder(encoder_out)

        return decoder_out, encoder_lens

    def apply_init(self, input_x, input_length, init=None):
        self.forward(input_x, input_length)

        if init is not None:
            self.apply(init)


def initialize_weights(module):
    if isinstance(module, (nn.Conv1d, nn.LazyConv1d, nn.Linear, nn.LazyLinear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
