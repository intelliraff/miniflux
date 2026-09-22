"""Hybrid MiniFlux with direct per-block CIFAR class modulation."""
import torch
from torch import nn
from torch.nn import functional as F

from models.cifar_hybrid_miniflux import CifarHybridMiniFlux
from models.mini_flux_v2 import ModulatedBlock


class ClassResidualBlock(nn.Module):
    def __init__(self, channels, condition_dim, class_rank=32):
        super().__init__()
        self.norm1=nn.GroupNorm(8,channels)
        self.conv1=nn.Conv2d(channels,channels,3,padding=1)
        self.norm2=nn.GroupNorm(8,channels)
        self.conv2=nn.Conv2d(channels,channels,3,padding=1)
        self.time_modulation=nn.Sequential(nn.SiLU(),nn.Linear(condition_dim,2*channels))
        self.class_modulation=nn.Sequential(nn.LayerNorm(condition_dim),nn.Linear(condition_dim,class_rank),nn.SiLU(),nn.Linear(class_rank,2*channels))
        self.activation=nn.SiLU()
        nn.init.normal_(self.class_modulation[-1].weight,std=.01)
        nn.init.zeros_(self.class_modulation[-1].bias)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)

    def forward(self,x,time_condition,class_condition):
        modulation=self.time_modulation(time_condition)+self.class_modulation(class_condition)
        shift,scale=modulation.unsqueeze(-1).unsqueeze(-1).chunk(2,dim=1)
        h=self.norm1(x)*(1+scale)+shift
        h=self.conv1(self.activation(h))
        h=self.conv2(self.activation(self.norm2(h)))
        return x+h


class ClassModulatedBlock(ModulatedBlock):
    def __init__(self,dim,heads,class_rank=32):
        super().__init__(dim,heads)
        self.class_modulation=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,class_rank),nn.SiLU(),nn.Linear(class_rank,6*dim))
        nn.init.normal_(self.class_modulation[-1].weight,std=.01)
        nn.init.zeros_(self.class_modulation[-1].bias)

    def forward(self,x,time_condition,class_condition):
        modulation=self.modulation(time_condition)+self.class_modulation(class_condition)
        shift1,scale1,gate1,shift2,scale2,gate2=modulation.unsqueeze(1).chunk(6,dim=-1)
        h=self.norm1(x)*(1+scale1)+shift1
        b,n,d=h.shape
        q,k,v=self.qkv(h).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        q=F.rms_norm(q,(d//self.heads,)); k=F.rms_norm(k,(d//self.heads,))
        h=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(b,n,d)
        x=x+gate1*self.out(h)
        return x+gate2*self.mlp(self.norm2(x)*(1+scale2)+shift2)


class StrongConditionedCifarHybridMiniFlux(CifarHybridMiniFlux):
    """Condition class identity directly into every residual and AdaLN block."""
    NUM_CLASSES=10
    NULL_CLASS=10

    def __init__(self,local_dim=64,base_dim=128,bottleneck_dim=256,bottleneck_depth=5,heads=4,dropout=.1,class_rank=32):
        super().__init__(local_dim,base_dim,bottleneck_dim,bottleneck_depth,heads,dropout)
        self.class_embedding=nn.Embedding(self.NUM_CLASSES+1,bottleneck_dim)
        self.class_projection=nn.Sequential(nn.LayerNorm(bottleneck_dim),nn.Linear(bottleneck_dim,bottleneck_dim),nn.SiLU())
        self.local_encoder=ClassResidualBlock(local_dim,bottleneck_dim,class_rank)
        self.base_encoder=ClassResidualBlock(base_dim,bottleneck_dim,class_rank)
        self.base_decoder=ClassResidualBlock(base_dim,bottleneck_dim,class_rank)
        self.local_decoder=ClassResidualBlock(local_dim,bottleneck_dim,class_rank)
        self.transformer=nn.ModuleList([ClassModulatedBlock(bottleneck_dim,heads,class_rank) for _ in range(bottleneck_depth)])
        for block in self.transformer:
            block.mlp=nn.Sequential(block.mlp[0],block.mlp[1],nn.Dropout(dropout),block.mlp[2])
        nn.init.normal_(self.class_embedding.weight,std=.02)

    def forward(self,x,timesteps,labels=None,extra=None):
        del extra
        if labels is None:
            labels=torch.full((x.shape[0],),self.NULL_CLASS,device=x.device,dtype=torch.long)
        labels=torch.as_tensor(labels,device=x.device,dtype=torch.long)
        if labels.shape!=(x.shape[0],): raise ValueError('labels must have shape (batch,)')
        if ((labels<0)|(labels>self.NULL_CLASS)).any(): raise ValueError('invalid CIFAR class label')
        time_condition=self.time_embedding(timesteps*1000.)
        class_condition=self.class_projection(self.class_embedding(labels))
        return self.forward_separate_conditions(x,time_condition,class_condition)

    def forward_separate_conditions(self,x,time_condition,class_condition):
        from models.cifar_hierarchical_miniflux import _feature_map,_spatial_positions,_tokens
        _,channels,height,width=x.shape
        if channels!=3 or height%4 or width%4: raise ValueError('requires RGB dimensions divisible by four')
        local=self.local_encoder(self.input_projection(x),time_condition,class_condition)
        base=self.base_encoder(self.down_to_base(local),time_condition,class_condition)
        low=self.down_to_bottleneck(base); low_h,low_w=height//4,width//4
        tokens=_tokens(low)+_spatial_positions(low_h,low_w,self.bottleneck_dim,x.device,low.dtype)
        for block in self.transformer: tokens=block(tokens,time_condition,class_condition)
        low=_feature_map(tokens,low_h,low_w)
        decoded_base=self.base_fusion(torch.cat((self.up_to_base(low),base),dim=1))
        decoded_base=self.base_decoder(decoded_base,time_condition,class_condition)
        decoded_local=self.local_fusion(torch.cat((self.up_to_local(decoded_base),local),dim=1))
        decoded_local=self.local_decoder(decoded_local,time_condition,class_condition)
        return self.output(decoded_local)
