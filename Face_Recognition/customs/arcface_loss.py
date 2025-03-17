import torch.nn as nn 
import torch 
import math
import torch.nn.functional as F

class Arcface(nn.Module):
    def __init__(self, margin=0.5, scale=30.0):
        super().__init__()

        self.margin = margin
        self.scale = scale

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)


    def forward(self, cosine_similarity, labels):
        sine = torch.sqrt(1.0 - torch.pow(cosine_similarity, 2))
        phi = cosine_similarity * self.cos_m - sine * self.sin_m

        # easy margin correction
        phi = torch.where(cosine_similarity > 0, phi, cosine_similarity)

        one_hot = torch.zeros_like(cosine_similarity)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        output = one_hot * phi + (1.0 - one_hot) * cosine_similarity
        output *= self.scale

        loss = F.cross_entropy(output, labels)

        return loss 



