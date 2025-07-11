import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
import math
from einops.layers.torch import Rearrange
from typing import Optional
from tqdm import tqdm
from time import time

#Transformer Engine-based Modules

class SwishFFN(nn.Module):
    def __init__(
            self,
            model_dim: int,
            ffn_dim: int,
            normalize: bool = False,
    ) -> None:
        super().__init__()

        self.ffn = nn.Sequential(
            te.LayerNorm(normalized_shape=model_dim) if normalize else nn.Identity(),
            te.Linear(in_features=model_dim, out_features=ffn_dim),
            nn.SiLU(),
            te.Linear(in_features=ffn_dim, out_features=model_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    
class GaussianFFN(nn.Module):
    def __init__(
            self,
            model_dim: int,
            ffn_dim: int,
            normalize: bool = False,
    ) -> None:
        super().__init__()

        self.ffn = nn.Sequential(
            te.LayerNorm(normalized_shape=model_dim) if normalize else nn.Identity(),
            te.Linear(in_features=model_dim, out_features=ffn_dim),
            nn.GELU(approximate='tanh'),
            te.Linear(in_features=ffn_dim, out_features=model_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class SelfAttention(nn.Module):
    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        n_heads: int,
        p_drop: float,
        normalize: bool = False,
        scale: str = 'identity'
    ) -> None:
        super().__init__()

        scale_factors = {
            'identity' : 1,
            'inv_sqrt': math.sqrt(inner_dim // n_heads),
        }
        self.scale = scale_factors.get(scale, 1.)

        self.rearrange = Rearrange('b n (h d) -> b n h d', h=n_heads)

        if normalize:
            self.ln_qkv_proj = te.LayerNormLinear(in_features=model_dim, out_features=3*inner_dim, bias=False)
        else:
            self.ln_qkv_proj = te.Linear(in_features=model_dim, out_features=3*inner_dim, bias=False)

        self.dropout = nn.Dropout(p=p_drop) if p_drop > 0 else nn.Identity()
        self.rearrange2 = Rearrange('b n h d -> b n (h d)')
        self.proj = te.Linear(in_features=inner_dim, out_features=model_dim, bias=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qkv = self.ln_qkv_proj(x)
        qkv = self.rearrange(qkv)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)

        scores = q @ k.transpose(-1, -2)
        scores /= self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e6)
        scores = self.dropout(scores)
        scores = F.softmax(scores, dim=-1)

        x = scores @ v

        x = self.rearrange2(x)
        return self.proj(x)

class EncoderBlock(nn.Module):
    def __init__(
        self, 
        model_dim: int, 
        inner_dim: int, 
        ffn_dim: int, 
        n_heads: int,
        p_drop: float = 0.,
        normalize: bool = False,
        scale: str = 'identity',
    ) -> None:
        super().__init__()

        self.sa = SelfAttention(model_dim=model_dim, inner_dim=inner_dim, n_heads=n_heads, p_drop=p_drop, normalize=normalize, scale=scale)
        #self.ffn = SwishFFN(model_dim=model_dim, ffn_dim=ffn_dim, normalize=normalize)
        self.ffn = GaussianFFN(model_dim=model_dim, ffn_dim=ffn_dim, normalize=normalize)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(x)
        return x + self.ffn(x)


#Inverted Bottleneck blocks

class InvertedResidualBottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inv_multiplier: int,
        dwconv_ker: int
    ) -> None:
        super().__init__()

        dwconv_pad = (dwconv_ker -1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim * inv_multiplier, kernel_size=1),
            nn.BatchNorm2d(num_features=input_dim * inv_multiplier),
            nn.SiLU()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim * inv_multiplier, out_channels=input_dim * inv_multiplier, kernel_size=dwconv_ker, padding=dwconv_pad, groups=input_dim * inv_multiplier),
            nn.BatchNorm2d(num_features=input_dim * inv_multiplier),
            nn.SiLU()
        )
        self.squeeze_and_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('n c h w -> n h w c'),
            te.Linear(in_features=input_dim * inv_multiplier, out_features=input_dim),
            nn.SiLU(),
            te.Linear(in_features=input_dim, out_features=input_dim * inv_multiplier),
            Rearrange('n h w c -> n c h w'),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=input_dim * inv_multiplier, out_channels=input_dim, kernel_size=1),
            nn.BatchNorm2d(num_features=input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.dwconv(x)
        x = x * self.squeeze_and_excite(x)
        return residual + self.proj(x)
    
class InvertedStridedBottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        inv_multiplier: int,
        output_dim: int,
        dwconv_ker: int,
        pool: bool = True
    ) -> None:
        super().__init__()

        dwconv_pad = (dwconv_ker -1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim * inv_multiplier, kernel_size=1),
            nn.BatchNorm2d(num_features=input_dim * inv_multiplier),
            nn.SiLU()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels=input_dim * inv_multiplier, out_channels=input_dim * inv_multiplier, kernel_size=dwconv_ker, padding=dwconv_pad, groups=input_dim * inv_multiplier),
            nn.BatchNorm2d(num_features=input_dim * inv_multiplier),
            nn.SiLU()
        )
        self.squeeze_and_excite = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('n c h w -> n h w c'),
            te.Linear(in_features=input_dim * inv_multiplier, out_features=input_dim),
            nn.SiLU(),
            te.Linear(in_features=input_dim, out_features=input_dim * inv_multiplier),
            Rearrange('n h w c -> n c h w'),
            nn.Sigmoid(),
        )
        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=input_dim * inv_multiplier, out_channels=output_dim, kernel_size=1),
            nn.BatchNorm2d(num_features=output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.dwconv(x)
        x = x * self.squeeze_and_excite(x)
        x = self.mpool(x)
        return self.proj(x)

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        inout_dim: int,
        inv_multiplier: int,
        dwconv_ker: int,
        normalize: bool = True, #This parameter is ignored as of current implementation.
        use_gelu: bool = False,
        p_stochastic_depth: float = 0.
    ) -> None:
        super().__init__()

        dwconv_pad = (dwconv_ker - 1) // 2
        self.dwconv = nn.Conv2d(
            in_channels=inout_dim, 
            out_channels=inout_dim, 
            kernel_size=dwconv_ker, 
            padding=dwconv_pad, 
            groups=inout_dim
        )
        '''self.channel_norm = nn.Sequential(
            Rearrange('n c h w -> n h w c'),
            te.LayerNorm(normalized_shape=inout_dim) if normalize else nn.Identity(),
        )
        self.pwconv = nn.Sequential(
            te.Linear(in_features=inout_dim, out_features=inout_dim * inv_multiplier),
            nn.GELU() if use_gelu else nn.SiLU(),
            te.Linear(in_features=inout_dim * inv_multiplier, out_features=inout_dim),
            Rearrange('n h w c -> n c h w'),
        )'''
        self.fusedpwconv = nn.Sequential(
            Rearrange('n c h w -> n h w c'),
            te.LayerNormMLP(
                hidden_size=inout_dim, 
                ffn_hidden_size=inout_dim * inv_multiplier, 
                activation='gelu' if use_gelu else 'relu',
                normalization='LayerNorm',
            ),
            Rearrange('n h w c -> n c h w'),
        )

        self.p_stochastic_depth = p_stochastic_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:       
        residual = x
        
        x = self.dwconv(x)
        '''x = self.channel_norm(x)
        x = self.pwconv(x)'''
        x = self.fusedpwconv(x)

        if self.training and self.p_stochastic_depth > 0:
            batch_size = x.size(0)
            mask = torch.rand((batch_size, 1, 1, 1)) > self.p_stochastic_depth
            mask = mask.cuda()
            survival_probability = 1 - self.p_stochastic_depth
            x = (x / survival_probability) * mask

        return x + residual

if __name__ == '__main__':
    enc = EncoderBlock(model_dim=768, inner_dim=768, ffn_dim=3072, n_heads=12, p_drop=0.1, normalize=True, scale='inv_sqrt').cuda()
    #enc = torch.compile(enc)

    torch.set_float32_matmul_precision('high')

    num_runs = 100  # Number of times to run
    times = []

    x = torch.randn(1, 196, 768).cuda()
    #x = torch.randn(1, 3, 224, 224)
    #y = enc(x)

    for _ in tqdm(range(num_runs), desc="Running Benchmark"):
        torch.cuda.synchronize()
        start_time = time()
        
        y = enc(x)
        
        torch.cuda.synchronize()
        end_time = time()
        
        times.append(end_time - start_time)

    avg_time = sum(times) / num_runs
    print(f"\nAverage Inference Time: {avg_time * 1e3 :.2f} ms over {num_runs} runs")
    print(f"Output Shape: {y.shape}") #type: ignore