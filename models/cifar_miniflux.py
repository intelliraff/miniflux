"""Unconditional pixel-space adapter of MiniFlux-v2 for the CIFAR benchmark."""
import torch
from torch import nn
from models.mini_flux import TimeEmbedding
from models.mini_flux_v2 import ModulatedBlock, sinusoidal_positions


class CifarMiniFlux(nn.Module):
    def __init__(self, hidden_dim=256, depth=5, heads=4, dropout=.3):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.image_projection=nn.Conv2d(3,hidden_dim,2,stride=2)
        self.time_embedding=TimeEmbedding(hidden_dim)
        self.blocks=nn.ModuleList([ModulatedBlock(hidden_dim,heads) for _ in range(depth)])
        for block in self.blocks:
            block.mlp=nn.Sequential(block.mlp[0],block.mlp[1],nn.Dropout(dropout),block.mlp[2])
        self.final_norm=nn.LayerNorm(hidden_dim,elementwise_affine=False)
        self.final_modulation=nn.Sequential(nn.SiLU(),nn.Linear(hidden_dim,2*hidden_dim))
        self.final=nn.Linear(hidden_dim,12)
        for layer in (self.final_modulation[-1],self.final):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self,x,timesteps,extra=None):
        b,c,h,w=x.shape
        if c!=3 or h%2 or w%2: raise ValueError('Requires RGB with even spatial dimensions')
        gh,gw=h//2,w//2
        rows,cols=torch.meshgrid(torch.arange(gh,device=x.device),torch.arange(gw,device=x.device),indexing='ij')
        pos=torch.cat([sinusoidal_positions(rows.flatten(),self.hidden_dim//2),sinusoidal_positions(cols.flatten(),self.hidden_dim//2)],dim=-1)
        x=self.image_projection(x).flatten(2).transpose(1,2)+pos
        condition=self.time_embedding(timesteps*1000.)
        for block in self.blocks: x=block(x,condition)
        shift,scale=self.final_modulation(condition).unsqueeze(1).chunk(2,dim=-1)
        x=self.final(self.final_norm(x)*(1+scale)+shift)
        return x.reshape(b,gh,gw,2,2,3).permute(0,5,1,3,2,4).reshape(b,3,h,w)
