import torch 
import torch.nn as nn 
from torchsummaryX import summary

class Patches(nn.Module):
    def __init__(self, img_size, embed, patch_size=(7, 7)):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed = embed

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "Image size must be divisible by patch size"

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.project = nn.Linear(patch_size[0] * patch_size[1], embed)

    def forward(self, x):
        # shape of x: (batch_size, time, features)
        batch_size, _, _ = x.shape 

        x = x.unfold(1, self.patch_size[0], self.patch_size[0])
        x = x.unfold(2, self.patch_size[1], self.patch_size[1])
        x = x.contiguous().view(batch_size, self.num_patches, self.patch_size[0] * self.patch_size[1])

        # shape of x: (batch_size, num_patches, patch_size)
        x = self.project(x) 

        # shape of x: (batch_size, num_patches, emb_size)
        return x


class MixerBlock(nn.Module):
    def __init__(self, num_patches, embed_dim, num_hidden_patches, num_hidden_emb, dropout):
        super().__init__()

        # input and output data shape: (batch_size, emb_size, num_patches)
        self.token_mixing = nn.Sequential(
            nn.LazyLinear(num_hidden_patches), 
            nn.Dropout(dropout),
            nn.GELU(), 
            nn.LazyLinear(num_patches), 
            nn.Dropout(dropout)
        )

        # input and output data shape: (batch_size, num_patches, emb_size)
        self.channel_mixing = nn.Sequential(
            nn.LazyLinear(num_hidden_emb), 
            nn.Dropout(dropout),
            nn.GELU(), 
            nn.LazyLinear(embed_dim), 
            nn.Dropout(dropout)
        )

        self.LN_token = nn.LayerNorm(embed_dim)
        self.LN_channel = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # shape of x: (batch_size, num_patches, emb_size)
        y = self.LN_token(x)
        y = y.transpose(1, 2)   # y shape: (batch_size, emb_size, num_patches)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.LN_channel(x)
        y = self.channel_mixing(y)
        x = x + y 

        return x

class MixerMLP(nn.Module):
    def __init__(self, num_inputs_time, num_inputs_features, embed_size, num_hidden_token, num_hidden_channel, num_outputs, num_blocks, dropout, patch_size=(7, 7)):
        super().__init__()

        self.patch_emb = Patches((num_inputs_time, num_inputs_features), embed_size, patch_size)
        self.num_patches = self.patch_emb.num_patches

        blk = []
        for _ in range(num_blocks):
            blk.append(MixerBlock(self.num_patches, embed_size, num_hidden_token, num_hidden_channel, dropout))

        self.net = nn.Sequential(*blk)
        self.LN = nn.LayerNorm(embed_size)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_size, num_outputs)
        )
        
    
    def forward(self, x):
        x = self.patch_emb(x)
        x = self.net(x)
        x = self.LN(x)
        x = x.mean(dim=1)

        return self.mlp_head(x)
        
    
    def apply_init(self, inputs, init=None):
        self.forward(inputs)

        if init is not None:
            self.net.apply(init)


def initialize_weights(module, init_method="xavier_normal"):
    if isinstance(module, nn.Linear):
        if init_method == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_method == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif init_method == "uniform":
            nn.init.uniform_(module.weight)
        else:
            raise ValueError(f"Invalid weight_initialization value: {init_method}")
        
        # Initialize bias to zeros if it exists
        if module.bias is not None:
            nn.init.zeros_(module.bias)

if __name__ == "__main__":
    model = MixerMLP(63, 28, 218, 512, 512, 42, 2, 0.1).to("mps")
    frames = torch.randn(2048, 63, 28)
    model.apply_init(frames.to("mps"), initialize_weights)
    summary(model, frames.to("mps"))
    

