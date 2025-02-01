import torch.nn as nn
import torch
from torchsummaryX import summary

class Res(nn.Module):
    def __init__(self, num_hiddens, dropout):
        super().__init__()

        self.res = nn.Sequential(
            nn.LazyLinear(num_hiddens), 
            nn.LazyBatchNorm1d(),
            nn.Dropout(dropout), 
            nn.GELU(), 
            nn.LazyLinear(num_hiddens), 
            nn.LazyBatchNorm1d(),
            nn.Dropout(dropout)
        )

        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.res(x))


class S_model(nn.Module):
    def __init__(self, num_blocks, num_hiddens, num_outputs, dropout):
        super().__init__()

        blk = []
        for _ in range(num_blocks):
            blk.append(Res(num_hiddens, dropout))
        self.blocks = nn.Sequential(*blk)

        self.mlp_head = nn.LazyLinear(num_outputs)

    def forward(self, x):
        # shape of x: (batch_size, time_step, features) --> (batch_size, time_step * features)
        x = torch.flatten(x, start_dim=1)
        return self.mlp_head(self.blocks(x))

if __name__ == "__main__":
    pass
