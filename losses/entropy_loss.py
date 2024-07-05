import torch
import torch.nn as nn
from torch.nn import functional as F

class EntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        '''
        x (attn_weights) : TxM
        '''
        loss = -1 * x * torch.log(x + self.eps)
        loss = torch.sum(loss, dim=-1)
        loss = torch.mean(loss)
        return loss