import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM
from perforatedai import pb_utils as PBU


class ResNetPB(nn.Module):
    def __init__(self, otherResNet):
        super(ResNetPB, self).__init__()
        
        self.DEVICE = otherResNet.DEVICE
        self.backbone = nn.Sequential()

        for id, layer in enumerate(otherResNet.backbone):
            if id == 0:
                continue
            
            if id == 1:
                self.backbone.add_module("first_two",
                    PBG.PBSequential((otherResNet.backbone[0], otherResNet.backbone[1]))
                )

            else:
                self.backbone.add_module(f"{id}",
                    otherResNet.backbone[id]
                )  

        self.cls_layer = otherResNet.cls_layer

    def forward(self, x):
        x = x.to(self.DEVICE)
        feats = self.backbone(x)
        out = self.cls_layer(feats)
        
        return {"feats": feats, "out": out}
    

class ResNeXtBlock(nn.Module):
    def __init__(self, intermediate_channels, out_channels, cardinality, downsample=False, stride=1):
        super().__init__()

        self.conv1 = nn.LazyConv2d(intermediate_channels, kernel_size=1)
        self.conv2 = nn.LazyConv2d(intermediate_channels, kernel_size=3, padding=1, stride=stride, groups= intermediate_channels // cardinality)
        self.conv3 = nn.LazyConv2d(out_channels, kernel_size=1)
        self.conv4 = nn.LazyConv2d(out_channels, kernel_size=1, stride=stride) if downsample else None

        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        self.bn4 = nn.LazyBatchNorm2d() if downsample else None 
    
    def forward(self, X):
        Y = self.bn1(self.conv1(X))
        Y = F.relu(Y)

        Y = self.bn2(self.conv2(Y))
        Y = F.relu(Y)

        Y = self.bn3(self.conv3(Y))

        if self.conv4:
            X = self.bn4(self.conv4(X))
        
        return F.relu(Y + X)


class CosineLinear(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(num_outputs, num_inputs))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        x_norm = F.normalize(x, dim=1)
        W_norm = F.normalize(self.W, dim=1)

        return F.linear(x_norm, W_norm)


class ResNeXt(nn.Module):
    def __init__(self, params, cardinality, num_classes, DEVICE="cuda", cosine_similarity=False):
        '''
        Parameters
        ----------
        params: list of tuples (num_block, intermediate_channels, out_channels)
        '''
        super().__init__()

        self.DEVICE = DEVICE

        # first part: 64 channels, downsample by 4 
        self.backbone = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, padding=3, stride=2), 
            nn.LazyBatchNorm2d(), nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )

        # self.backbone = PBG.PBSequential((
        #     nn.LazyConv2d(64, kernel_size=7, padding=3, stride=2), 
        #     nn.LazyBatchNorm2d(), nn.ReLU(), 
        #     nn.MaxPool2d(kernel_size=3, padding=1, stride=2))
        # )


        # Add ResNeXt blocks 
        for i, param in enumerate(params):
            # do not downsample for the first block as it follows the maxpool 
            self.backbone.add_module(name=f'block{i+2}', module=self.block(param[0], cardinality, param[1], param[2], first_block=(i==0)))

        self.backbone.add_module(
            'global_pool',
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        )

        # output layer
        self.cls_layer= nn.Sequential(nn.LazyLinear(num_classes) if not cosine_similarity else CosineLinear(num_inputs=params[-1][-1], num_outputs=num_classes))
        
    def block(self, num_block, cardinality, intermediate_channels, out_channels, first_block=False):
        block = []

        for i in range(num_block):
            if i == 0 and not first_block:
                block.append(ResNeXtBlock(intermediate_channels, out_channels, cardinality, downsample=True, stride=2))
            elif i == 0:
                block.append(ResNeXtBlock(intermediate_channels, out_channels, cardinality, downsample=True))
            else:
                block.append(ResNeXtBlock(intermediate_channels, out_channels, cardinality))

        return nn.Sequential(*block)

    def apply_init(self, inputs, init=None):
        self.forward(inputs)

        if init is not None:
            self.apply(init)

    def forward(self, x):
        x = x.to(self.DEVICE)
        feats = self.backbone(x)
        out = self.cls_layer(feats)
        
        return {"feats": feats, "out": out}


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.LazyConv2d, nn.Linear, nn.LazyLinear)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


        
if __name__ == "__main__":
    model = ResNeXt(params=[(3, 64, 64), (4, 64, 96), 
                            (6, 96, 192), (3, 192, 384)],
                            cardinality=4, 
                            num_classes=8631).to("cuda")
    
    summary(model, (3, 112, 112))
