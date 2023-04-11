import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange 

class feedforward( nn.Module ):
    def __init__(self, dim, hidden_dim, dropout=0 ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear( dim ,hidden_dim ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear( hidden_dim, dim ),
            nn.Dropout(dropout)
        )
    
    def forward(self , x):
        return self.net(x)
        
class Patch_Pos_embed( nn.Module ):
    def __init__(self, dim, patch_len, pos_num, channels:int = 3, patch_size:int = 16, dropout=0 ):
        super().__init__()
        self.embed_dim  = dim
        self.patch_len  = patch_len
        
        self.dp         = nn.Dropout(dropout) 
        self.position   = nn.Parameter( torch.randn(1, (self.patch_len ** 2) + pos_num, self.embed_dim ) )
       
        self.projection = nn.Sequential(
            nn.Conv2d( channels, self.embed_dim ,kernel_size=patch_size , stride=patch_size , bias=False ),
            Rearrange('b e (h) (w) -> b (h w) e'),
            nn.LayerNorm(dim)
        )
         
    def forward(self , image):

        x = self.projection(image)
        b, n, _ = x.shape
        
        x += self.position[:, :n]
        x  = self.dp(x)
        return x , self.patch_len
    
class SRA( nn.Module ):
    def __init__(self, dim, heads, reduction, attn_dp=0, proj_dp=0 ) -> None:
        super().__init__()    
        
        self.heads   = heads
        self.scale   = (dim //  heads) ** -0.5
        self.redc    = reduction
        self.to_q      = nn.Linear(dim, dim  , bias = False)
        self.to_kv     = nn.Linear(dim, dim*2, bias = False)
        self.softmax   = nn.Softmax(dim = -1)
        self.attn_drop = nn.Dropout(attn_dp)
        
        self.SR = nn.Sequential(
            nn.Conv2d( dim, dim, kernel_size=reduction, stride=reduction ),
            Rearrange( 'b c h w -> b (h w) c' ),
            nn.LayerNorm( dim )
        )
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dp)
        )
        self.norm = nn.LayerNorm( dim )
        
    def forward( self, x, H, W ):
        B, N, C = x.shape
        q = rearrange( self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        if self.redc > 1:
            x_   = rearrange( x, ' b (h w) c -> b c h w', h=H, w=W )
            redc = self.SR(x_)
            kv  =  self.to_kv(redc).chunk(2,dim = -1)
            k , v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        else:
            kv  =  self.to_kv(x).chunk(2,dim = -1)
            k , v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        
        attn = ( q @ k.transpose(-1, -2) ) * self.scale
        attn = self.softmax( attn )
        attn = self.attn_drop(attn)
        
        out = (attn @ v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out

class Encoder( nn.Module ):
    def __init__(self, dim, num_heads , reduction, mlp_ratio, attn_dp=0, proj_dp=0, ff_dp=0 ):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.sra = SRA(dim=dim, heads=num_heads, reduction=reduction, attn_dp=attn_dp, proj_dp=proj_dp )
        hidden_dim = dim * mlp_ratio
        self.ffn = feedforward(dim=dim, hidden_dim=hidden_dim, dropout=ff_dp)

    def forward(self, x, h, w):
        
        x_norm    = self.norm(x)
        attn      = x + self.sra(x_norm,H=h, W=w)
        attn_norm = self.norm(attn)
        ff        = attn + self.ffn(attn_norm)
        
        return ff

class block( nn.Module ):
    def __init__(self,dim, num_heads, reduction, depth, mlp_ratio, attn_dp, proj_dp=0, ff_dp=0 ) :
        super().__init__()
        self.norm    = nn.LayerNorm( dim ) 
        self.Elayer  = nn.ModuleList([]) 
        
        self.Elayer.append(nn.ModuleList([
            Encoder( dim=dim, num_heads=num_heads, reduction=reduction, mlp_ratio=mlp_ratio, attn_dp=attn_dp, proj_dp=proj_dp, ff_dp=ff_dp ) 
            for _ in range(depth)])
        )

    def forward(self, x, h, w):
        
        for i,ec in enumerate( self.Elayer):
            x = ec[i](x, h, w)

        return x

class PVT( nn.Module ):
    def __init__(self, img_size=224, patch_size=16, classes=500, embed_dim=[64,128,256,512], num_heads=[1,2,4,8],
                 mlp_ratio = [4,4,4,4], qkv_bias=False, drop=0., attn_drop=0, block_depth=[3,4,6,3], sr_ratio=[8,4,2,1],
                 num_stage=4):
        super().__init__()
        self.classes     = classes
        self.block_depth = block_depth
        self.num_stages  = num_stage

        self.Elayers = nn.ModuleList([])
        for i in range(num_stage):
            
            size = img_size if i == 0 else img_size // (2 ** (i + 1)) 
            patch_size = patch_size if i == 0 else 2
            patch_len = ( size // patch_size )

            self.Elayers.append(nn.ModuleList([
                Patch_Pos_embed( dim=embed_dim[i], 
                                 patch_len=patch_len ,
                                 pos_num=0 if i != num_stage-1 else 1 ,
                                 channels=3 if i == 0 else embed_dim[i-1],
                                 patch_size=patch_size), 
                block( dim=embed_dim[i], num_heads=num_heads[i], reduction=sr_ratio[i], depth=block_depth[i], 
                      mlp_ratio=mlp_ratio[i], attn_dp=attn_drop, proj_dp=0, ff_dp=0 )
            ]))
        self.norm = nn.LayerNorm(embed_dim[3])
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim[3]))
        self.classifier = nn.Sequential(
            nn.LayerNorm( embed_dim[3] ),
            nn.Linear( embed_dim[3] , self.classes )
        )
        
    def forward( self, image ):
        b = image.shape[0]
        x = image
        stage = 0
        for pp, blks in  self.Elayers:
            
            x , patch_len = pp(x)
            if stage == self.num_stages-1:
                x             = self.norm(x)
                cls_tokens    = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
                x             = torch.cat((cls_tokens, x), dim=1)
            x             = blks(x, patch_len, patch_len)

            if stage != self.num_stages-1:
                x = rearrange(x, 'b (h w ) c -> b c h w ', h=patch_len, w=patch_len)
            stage += 1
        x = x[:,0]
        x = self.classifier(x)
        
        return x
        
        
        
