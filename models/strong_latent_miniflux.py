"""Strongly text-conditioned MiniFlux operating in frozen VAE latent space."""
import torch
from torch import nn
from torch.nn import functional as F

from models.mini_flux import TimeEmbedding
from models.mini_flux_v2 import sinusoidal_positions


class TextModulatedBlock(nn.Module):
    def __init__(self,dim=256,heads=4,condition_rank=32,dropout=.1):
        super().__init__(); self.heads=heads
        self.norm1=nn.LayerNorm(dim,elementwise_affine=False); self.norm2=nn.LayerNorm(dim,elementwise_affine=False)
        self.qkv=nn.Linear(dim,3*dim); self.out=nn.Linear(dim,dim)
        self.mlp=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(approximate='tanh'),nn.Dropout(dropout),nn.Linear(4*dim,dim))
        self.time_modulation=nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        self.text_modulation=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,condition_rank),nn.SiLU(),nn.Linear(condition_rank,6*dim))
        nn.init.zeros_(self.time_modulation[-1].weight); nn.init.zeros_(self.time_modulation[-1].bias)
        nn.init.normal_(self.text_modulation[-1].weight,std=.01); nn.init.zeros_(self.text_modulation[-1].bias)

    def forward(self,x,time_condition,text_condition):
        modulation=self.time_modulation(time_condition)+self.text_modulation(text_condition)
        shift1,scale1,gate1,shift2,scale2,gate2=modulation.unsqueeze(1).chunk(6,dim=-1)
        h=self.norm1(x)*(1+scale1)+shift1; b,n,d=h.shape
        q,k,v=self.qkv(h).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        q=F.rms_norm(q,(d//self.heads,)); k=F.rms_norm(k,(d//self.heads,))
        h=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,d)
        x=x+gate1*self.out(h)
        return x+gate2*self.mlp(self.norm2(x)*(1+scale2)+shift2)


class StrongLatentMiniFlux(nn.Module):
    def __init__(self,latent_channels=4,text_dim=512,hidden_dim=256,depth=6,heads=4,condition_rank=32,dropout=.1,time_scale=1000.):
        super().__init__()
        if hidden_dim%4 or hidden_dim%heads: raise ValueError('hidden_dim must be divisible by four and heads')
        self.time_scale=time_scale; self.hidden_dim=hidden_dim; self.latent_channels=latent_channels
        self.image_projection=nn.Conv2d(latent_channels,hidden_dim,2,stride=2)
        self.text_projection=nn.Linear(text_dim,hidden_dim)
        self.time_embedding=TimeEmbedding(hidden_dim)
        self.text_condition=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.SiLU())
        self.blocks=nn.ModuleList([TextModulatedBlock(hidden_dim,heads,condition_rank,dropout) for _ in range(depth)])
        self.final_norm=nn.LayerNorm(hidden_dim,elementwise_affine=False)
        self.final_time=nn.Sequential(nn.SiLU(),nn.Linear(hidden_dim,2*hidden_dim))
        self.final_text=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,condition_rank),nn.SiLU(),nn.Linear(condition_rank,2*hidden_dim))
        self.final=nn.Linear(hidden_dim,latent_channels*4)
        nn.init.zeros_(self.final_time[-1].weight); nn.init.zeros_(self.final_time[-1].bias)
        nn.init.normal_(self.final_text[-1].weight,std=.01); nn.init.zeros_(self.final_text[-1].bias)
        nn.init.zeros_(self.final.weight); nn.init.zeros_(self.final.bias)

    def forward(self,latent,text,timestep):
        b,_,h,w=latent.shape
        if h%2 or w%2: raise ValueError('latent spatial dimensions must be even')
        gh,gw=h//2,w//2
        image=self.image_projection(latent).flatten(2).transpose(1,2)
        rows,cols=torch.meshgrid(torch.arange(gh,device=latent.device),torch.arange(gw,device=latent.device),indexing='ij')
        image=image+torch.cat((sinusoidal_positions(rows.flatten(),self.hidden_dim//2),sinusoidal_positions(cols.flatten(),self.hidden_dim//2)),dim=-1).to(image.dtype)
        text=self.text_projection(text)
        text_condition=self.text_condition(text.mean(dim=1))
        text=text+sinusoidal_positions(torch.arange(text.shape[1],device=text.device),self.hidden_dim).to(text.dtype)
        x=torch.cat((text,image),dim=1); time_condition=self.time_embedding(timestep*self.time_scale)
        for block in self.blocks: x=block(x,time_condition,text_condition)
        x=x[:,text.shape[1]:]
        shift,scale=(self.final_time(time_condition)+self.final_text(text_condition)).unsqueeze(1).chunk(2,dim=-1)
        x=self.final(self.final_norm(x)*(1+scale)+shift)
        return x.reshape(b,gh,gw,2,2,self.latent_channels).permute(0,5,1,3,2,4).reshape(b,self.latent_channels,h,w)
