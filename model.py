import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from experimental_modules import InvertedResidualBottleneck, InvertedStridedBottleneck, ConvNeXtBlock
from calflops import calculate_flops
from einops.layers.torch import Rearrange
from typing import List

class EfficientNet(nn.Module):
    def __init__(self, 
                n_classes: int,
                channel_dims: List[int],
                layer_numbers: List[int],
                dwconv_kernels: List[int],
                spatial_reductions: List[bool],
                stem_dim: int,
                feature_channels: int):
        super().__init__()

        self.n_classes = n_classes
        self.channel_dims = channel_dims
        self.layer_numbers = layer_numbers
        self.feature_channels = feature_channels
        self.dwconv_kernels = dwconv_kernels
        self.spatial_reductions = spatial_reductions
        self.stem_dim = stem_dim

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.stem_dim),
            nn.SiLU()
        )

        self.convolution_modules = nn.ModuleList([
            self._make_stage(
                input_dim=self.stem_dim if i == 0 else self.channel_dims[i-1],
                output_dim=self.channel_dims[i],
                num_layers=self.layer_numbers[i],
                spatial_reduction=self.spatial_reductions[i],
                dwconv_ker=self.dwconv_kernels[i],
                inv_multiplier=1 if i == 0 else 6
            ) for i in range(len(self.layer_numbers))
        ])

        self.conv_head = nn.Sequential(
            nn.Conv2d(self.channel_dims[-1], self.feature_channels, kernel_size=1),
            nn.BatchNorm2d(self.feature_channels),
            nn.SiLU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('n c h w -> n (c h w)'),
            nn.Linear(self.feature_channels, self.n_classes)
        )

        print(f'[EfficientNet] Model successfully initialized.')

    def _make_stage(self,
                    input_dim: int,
                    output_dim: int,
                    num_layers: int,
                    inv_multiplier: int,
                    spatial_reduction: bool,
                    dwconv_ker: int) -> nn.Module:
        
        def _layer_factory(layer_index: int) -> nn.Module:
            if layer_index == 0:
                return InvertedStridedBottleneck(
                    input_dim=input_dim,
                    inv_multiplier=inv_multiplier,
                    output_dim=output_dim,
                    dwconv_ker=dwconv_ker,
                    pool=spatial_reduction
                )
            else:
                return InvertedResidualBottleneck(
                    input_dim=output_dim,
                    inv_multiplier=inv_multiplier,
                    dwconv_ker=dwconv_ker
                )
            
        return nn.Sequential(*[
            _layer_factory(i) for i in range(num_layers)
        ])

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for convolution_module in self.convolution_modules:
            x = convolution_module(x)
        x = self.conv_head(x)

        return self.classifier(x)

    def calculate_flops(self) -> None:
        self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)

        print(f'[EfficientNet] Parameters: {self.params / 1e6 : .2f}M, MACs: {self.macs / 1e9 : .4f}GMACs') #type: ignore

    
class ConvNeXt(nn.Module):
    def __init__(self, 
                channel_dims: List[int], 
                layer_numbers: List[int], 
                p_stochastic_depth: float, 
                stochastic_depth_linear_decay: bool,
                n_classes: int) -> None:
        super().__init__()
        
        self.layer_numbers = layer_numbers
        self.channel_dims = channel_dims
        self.p_stochastic_depth = p_stochastic_depth
        self.linear_decay = stochastic_depth_linear_decay
        self.n_classes = n_classes

        self.stem = nn.Conv2d(in_channels=3, out_channels=channel_dims[0], kernel_size=4, stride=4)

        self.convolutional_blocks = nn.ModuleList([
            self._make_stage(
                input_dim=self.channel_dims[i-1],
                stage_dim=self.channel_dims[i],
                num_layers=self.layer_numbers[i],
                p_stochastic_depth=self.p_stochastic_depth,
                accumulated_depth=sum(self.layer_numbers[:i+1])-self.layer_numbers[i],
                downsample=False if i ==0 else True,
            ) for i in range(len(self.layer_numbers))
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('n c h w -> n (c h w)'),
            nn.Linear(in_features=self.channel_dims[-1], out_features=n_classes)
        )

        print(f'[ConvNeXt] Model successfully initialized.')

    def _make_stage(self,
                    input_dim: int, 
                    stage_dim: int, 
                    num_layers: int, 
                    p_stochastic_depth: float, 
                    accumulated_depth: int,
                    downsample: bool) -> nn.Module:
        
        dsl = nn.Sequential(
            Rearrange('n c h w -> n h w c'),
            te.LayerNorm(normalized_shape=input_dim),
            Rearrange('n h w c -> n c h w'),
            nn.Conv2d(in_channels=input_dim, out_channels=stage_dim, kernel_size=2, stride=2),
        ) if downsample else nn.Identity()
        
        convnext_blocks = [
            ConvNeXtBlock(
                inout_dim=stage_dim, 
                inv_multiplier=4, 
                dwconv_ker=7, 
                normalize=True, 
                p_stochastic_depth=p_stochastic_depth * i / sum(self.layer_numbers) if self.linear_decay else p_stochastic_depth,
                use_gelu=True,
            )
            for i in range(accumulated_depth, accumulated_depth + num_layers)
        ]

        return nn.Sequential(dsl, *convnext_blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for convolutional_block in self.convolutional_blocks:
            x = convolutional_block(x)

        return self.classifier(x)
    
    def calculate_flops(self):
        self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)
        print(f'[ConvNeXt] Parameters: {self.params / 1e6 : .2f}M, MACs: {self.macs / 1e9 : .4f}GMACs') #type: ignore

if __name__ == '__main__':
    channel_dims = [96, 192, 384, 768]
    num_layers = [3, 3, 9, 3]
    convnext = ConvNeXt(channel_dims=channel_dims, layer_numbers=num_layers, p_stochastic_depth=0.5, stochastic_depth_linear_decay=True, n_classes=1000).cuda()
    convnext.calculate_flops()

    ema = ConvNeXt(channel_dims=channel_dims, layer_numbers=num_layers, p_stochastic_depth=0.5, stochastic_depth_linear_decay=True, n_classes=1000).cuda()

    with torch.no_grad():
        for ema_param, param in zip(ema.parameters(), convnext.parameters()):
            ema_param.data.copy_(param.data)

    channel_dims = [16, 24, 40, 80, 112, 192, 320]
    num_layers = [1, 2, 2, 3, 3, 4, 1]
    spatial_reductions = [False, True, True, False, True, True, False]
    dwconv_kernels = [3, 3, 5, 3, 5, 5, 3]
    efficientnet = EfficientNet(
        n_classes=1000, 
        channel_dims=channel_dims, 
        layer_numbers=num_layers, 
        dwconv_kernels=dwconv_kernels, 
        spatial_reductions=spatial_reductions, 
        stem_dim=32, 
        feature_channels=1280
    ).cuda()
    efficientnet.calculate_flops()
