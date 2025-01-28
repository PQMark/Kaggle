import torch 
import torch.nn as nn 
from torchsummary import summary

class MixerBlock(nn.Module):
    def __init__(self, num_inputs_time, num_intputs_features, num_hidden_time, num_hidden_features, dropout):
        super().__init__()

        # input and output data shape: (batch_size, features, time)
        self.time_mixing = nn.Sequential(
            nn.LazyLinear(num_hidden_time), 
            nn.Dropout(dropout),
            nn.GELU(), 
            nn.LazyLinear(num_inputs_time), 
            nn.Dropout(dropout)
        )

        # input and output data shape: (batch_size, time, features)
        self.feature_mixing = nn.Sequential(
            nn.LazyLinear(num_hidden_features), 
            nn.Dropout(dropout),
            nn.GELU(), 
            nn.LazyLinear(num_intputs_features), 
            nn.Dropout(dropout)
        )

        self.LN_time = nn.LayerNorm(num_intputs_features)
        self.LN_feature = nn.LayerNorm(num_intputs_features)
    
    def forward(self, x):
        # shape of x: (batch_size, time, features)
        y = self.LN_time(x)
        y = y.transpose(1, 2)   # y shape: (batch_size, features, time)
        y = self.time_mixing(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.LN_feature(x)
        y = self.feature_mixing(y)
        x = x + y 

        return x

class MixerMLP(nn.Module):
    def __init__(self, num_inputs_time, num_intputs_features, num_hidden_time, num_hidden_features, num_outputs, num_blocks, dropout=0.4):
        super().__init__()

        blk = []
        for _ in range(num_blocks):
            blk.append(MixerBlock(num_inputs_time, num_intputs_features, num_hidden_time, num_hidden_features, dropout))

        self.net = nn.Sequential(*blk)

        self.net.add_module("Last", nn.Sequential(
            nn.LazyLinear(num_outputs)
        ))
    
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = MixerMLP(61, 2800, 128, 3600, 36, 5)
    input_tensor = torch.randn(1, 61, 2800)
    summary(model, input_size=(61, 2800))

