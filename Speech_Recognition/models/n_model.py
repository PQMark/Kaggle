import torch.nn as nn
import torch.nn.functional as F 
import torch
from torchsummaryX import summary

class DenseNet(nn.Module):
    def __init__(self, arch, num_ouputs, dropout):
        super().__init__()

        if isinstance(dropout, (int, float)):
            dropout_list = [dropout] * len(arch)
        else:
            assert len(dropout) == len(arch), "Length of dropout list must equal number of layers in 'arch'"

        blk = []
        for num_hidden, drop in zip(arch, dropout_list):
            blk.append(nn.Sequential(
                nn.LazyLinear(num_hidden), 
                nn.LazyBatchNorm1d(), 
                nn.SiLU(),
                nn.Dropout(drop)
            ))
        
        blk.append(nn.Sequential(
            nn.LazyLinear(num_ouputs)
        ))
        
        self.net = nn.Sequential(*blk)
    

    def forward(self, x):
        
        x = torch.flatten(x, start_dim=1)

        return self.net(x)
    
    def apply_init(self, inputs, init=None):
        self.forward(inputs)

        if init is not None:
            self.apply(init)


if __name__ == "__main__":
    model = DenseNet((4096, 2048, 1024, 1024, 750, 512), 42, 0.2).to("mps")
    frames = torch.randn(2048, 63, 28).to("mps")
    summary(model, frames)