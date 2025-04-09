from models.ASR import Permute, pBLSTM, LSTMWrapper
import torch 
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import torch.nn as nn


class LockedDropout(torch.nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate

    def _lock_dropout(self, x):
        b, t, f = x.size()
        mask = x.new_empty(b, 1, f, requires_grad=False)
        mask.bernoulli_(1 - self.dropout_rate)
        mask.div_(1 - self.dropout_rate)
        return x * mask

    def forward(self, x):
        # No dropout is applied during evaluation
        if not self.training or self.dropout_rate == 0:
            return x

        if isinstance(x, PackedSequence):
            padded, lengths = pad_packed_sequence(x, batch_first=True)
            padded = self._lock_dropout(padded)
            return pack_padded_sequence(
                padded,
                lengths=lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
        
        return self._lock_dropout(x)


class TDNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding=0, dropout_rate=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.conv1d(inputs)
        x = self.batchnorm(x)
        x = self.activation(x)
        return self.dropout(x)


class TDNNResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout_rate=0.1):
        super().__init__()
        
        padding = (kernel_size // 2) * dilation

        self.tdnn1 = TDNNLayer(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding, 
            dropout_rate=dropout_rate, 
            dilation=dilation
        )

        self.tdnn2 = TDNNLayer(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dropout_rate=dropout_rate, 
            dilation=dilation
        )

        # to handle channel or stride mismatches
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.final_dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.tdnn1(x)
        out = self.tdnn2(out)
        out += self.shortcut(x)
        return self.final_dropout(out)


class ResidualPBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM with residual connections and layer normalization
    '''
    def __init__(self, input_size, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.blstm = nn.LSTM(
            input_size=2 * input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.need_projection = (input_size != hidden_size*2)
        if self.need_projection:
            self.projection = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_packed):
        padded, orig_lengths = pad_packed_sequence(x_packed, batch_first=True)

        x_reduced, reduced_lengths = self._trunc_reshape(padded, orig_lengths)
        x_reduced_packed = pack_padded_sequence(x_reduced, reduced_lengths, batch_first=True, enforce_sorted=False)

        # BiLSTM
        lstm_out_packed, _ = self.blstm(x_reduced_packed)
        lstm_out, lstm_lengths = pad_packed_sequence(lstm_out_packed, batch_first=True)
        
        residual = padded[:, :lstm_out.size(1) * 2:2, :]
        if self.need_projection:
            residual = self.projection(residual)

        out = self.dropout(self.layer_norm(lstm_out + residual))

        return pack_padded_sequence(out, lstm_lengths, batch_first=True, enforce_sorted=False)

    def _trunc_reshape(self, x, x_lens):
        batch_size, t, f_dim = x.size()

        if t % 2 != 0:
            x = x[:, :-1, :]
            t = t - 1

        new_t = t // 2

        # Reshape x
        x = x.reshape(batch_size, new_t, 2, f_dim)
        x = x.reshape(batch_size, new_t, 2 * f_dim)

        new_lens = [l // 2 for l in x_lens]
        new_lens = torch.tensor(new_lens, dtype=torch.long)

        return x, new_lens


class Encoder(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size, dropout_rate=0.3):
        super().__init__()

        self.conv_layer = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3, padding=1)
        self.conv_bn = nn.BatchNorm1d(128)
        self.conv_activation = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout_rate)

        # TDNN
        self.tdnn_block1 = TDNNResBlock(in_channels=128, out_channels=128, kernel_size=3, dilation=1, dropout_rate=dropout_rate)
        self.tdnn_block2 = TDNNResBlock(in_channels=128, out_channels=256, kernel_size=3, dilation=2, dropout_rate=dropout_rate)
        self.tdnn_block3 = TDNNResBlock(in_channels=256, out_channels=512, kernel_size=3, dilation=4, dropout_rate=dropout_rate)

        self.projector = nn.Conv1d(128, 512, kernel_size=1, bias=False)
        self.cross_dropout = nn.Dropout(dropout_rate / 2)

        # pBLSTM
        self.pBLSTM1 = ResidualPBLSTM(input_size=512, hidden_size=encoder_hidden_size, dropout_rate=0.2)
        self.locked_dropout1 = LockedDropout(0.3)
        self.pBLSTM2 = ResidualPBLSTM(input_size=encoder_hidden_size*2, hidden_size=encoder_hidden_size, dropout_rate=0.2)
        self.locked_dropout2 = LockedDropout(0.3)

        self.final_layer_norm = nn.LayerNorm(encoder_hidden_size * 2)
        self.final_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, lengths):
        x = x.transpose(1, 2)       # shape: (batch_size, time, f_dim) --> (batch_size, f_dim, time)

        # conv block
        x = self.conv_layer(x)
        x = self.conv_bn(x)
        x = self.conv_activation(x)
        x = self.conv_dropout(x)

        # TDNN
        out1 = self.tdnn_block1(x)
        out2 = self.tdnn_block2(out1)
        out3 = self.tdnn_block3(out2)

        cross_feature = self.projector(out1)
        cross_feature = self.cross_dropout(cross_feature)
        out3 = out3 + cross_feature

        x = out3.transpose(1, 2)      # shape: (batch_size, time, f_dim)

        # Pack the padded sequence
        packed_seq = pack_padded_sequence(x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)

        # pBLSTM layers
        pblstm_out1 = self.pBLSTM1(packed_seq)
        pblstm_out1 = self.locked_dropout1(pblstm_out1)
        pblstm_out2 = self.pBLSTM2(pblstm_out1)
        pblstm_out2 = self.locked_dropout2(pblstm_out2)

        unpacked, updated_lens = pad_packed_sequence(pblstm_out2, batch_first=True)
        normalized = self.final_layer_norm(unpacked)
        final_out = self.final_dropout(normalized)

        return final_out, updated_lens


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


class AASRModel(torch.nn.Module):
    def __init__(self, input_size, embed_size, linear_arch, decoder_dropout, output_size=41):
       super().__init__()

       # Initialize encoder and decoder
       self.encoder = Encoder(input_size=input_size, encoder_hidden_size=embed_size)
       self.decoder = Decoder(embed_size, linear_arch, decoder_dropout, output_size)

    def forward(self, x, lengths_x):
       encoder_out, encoder_lens = self.encoder(x, lengths_x)
       decoder_out               = self.decoder(encoder_out)
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

