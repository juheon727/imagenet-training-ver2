import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from einops.layers.torch import Rearrange
from typing import Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
from calflops import calculate_flops

torch.autograd.set_detect_anomaly(True)

class ResidualModule(nn.Module, ABC):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 stride: int,
                 skip_connection: Optional[nn.Module] = None,
                 p_stochastic_depth: float = 0.):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.skip_connection = skip_connection
        self.stride = stride
        self.p_stochastic_depth = p_stochastic_depth

        self.activation = nn.ReLU()
        self.convlayers = self._make_convlayers()

        self._initialize_weights()

    @abstractmethod
    def _make_convlayers(self) -> nn.ModuleList:
        '''Implement code for making convolutional layers'''
        pass

    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample Stochastic Depth that actually *skips* convolution for
        a random subset of samples in the batch.

        Training speed with Batch Size = 512 on ResNet-18 configuration using Adam optimizer: 4.68it/s
        Requires further improvement.
        """
        if self.p_stochastic_depth > 0 and self.training:
            batch_size = x.size(0)
            p_survival = 1 - self.p_stochastic_depth
            idx_to_survive = torch.rand((batch_size), device='cuda') < p_survival

            if self.skip_connection is not None:
                residual = self.skip_connection(x)
            else:
                residual = x.clone()

            out = residual.clone()
            if idx_to_survive.sum() > 0:
                out[idx_to_survive] = residual[idx_to_survive] + self._conv_forward(x[idx_to_survive])

            return out

        else:
            if self.skip_connection is not None:
                residual = self.skip_connection(x)
            else:
                residual = x.clone()
            
            x = self._conv_forward(x)
            return self.activation(x + residual)
    
    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.convlayers:
            x = conv_layer(x)
        if (not self.training) or (self.p_stochastic_depth == 0):
            x = x * (1 - self.p_stochastic_depth)

        return x
    '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A fallback implementation of Per-Sample SD with masking.
        Performs convolution on all samples.

        Training speed with Batch Size = 512 on ResNet-18 configuration using Adam optimizer: 4.92it/s
        (actually faster lol)
        """
        residual = x
        if self.skip_connection is not None:
            residual = self.skip_connection(residual)

        for conv_layer in self.convlayers:
            x = conv_layer(x)

        if self.training and self.p_stochastic_depth > 0:
            batch_size = x.size(0)
            mask = torch.rand((batch_size, 1, 1, 1), device='cuda') > self.p_stochastic_depth
            x = x * mask
        elif self.p_stochastic_depth > 0:
            x = x * (1 - self.p_stochastic_depth)

        return self.activation(x + residual)
    
    def _initialize_weights(self):
        '''Initialize weights for all layers in the module'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming/He initialization for ReLU-based networks
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BatchNorm weights to 1 and bias to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class BottleneckModule(ResidualModule):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 inner_scale: int = 4,
                 stride: int = 1,
                 skip_connection: Optional[nn.Module] = None,
                 p_stochastic_depth: float = 0.):
        self.inner_scale = inner_scale
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            skip_connection=skip_connection,
            stride=stride,
            p_stochastic_depth = p_stochastic_depth
        )

    def _make_convlayers(self):
        conv1x1a = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.output_dim // self.inner_scale, kernel_size=1),
            nn.BatchNorm2d(num_features=self.output_dim // self.inner_scale),
            nn.ReLU()
        )
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=self.output_dim // self.inner_scale, out_channels=self.output_dim // self.inner_scale, kernel_size=3, padding=1, stride=self.stride),
            nn.BatchNorm2d(num_features=self.output_dim // self.inner_scale),
            nn.ReLU(),
        )
        conv1x1b = nn.Sequential(
            nn.Conv2d(in_channels=self.output_dim // self.inner_scale, out_channels=self.output_dim, kernel_size=1),
            nn.BatchNorm2d(num_features=self.output_dim),
        )

        return nn.ModuleList([conv1x1a, conv3x3, conv1x1b])
    
class PreActivationBottleneckModule(ResidualModule):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 inner_scale: int = 4,
                 stride: int = 1,
                 skip_connection: Optional[nn.Module] = None,
                 p_stochastic_depth: float = 0.):
        self.inner_scale = inner_scale
        
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            skip_connection=skip_connection,
            stride=stride,
            p_stochastic_depth=p_stochastic_depth
        )

    def _make_convlayers(self):
        conv1x1a = nn.Sequential(
            nn.BatchNorm2d(num_features=self.input_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.output_dim // self.inner_scale, kernel_size=1),
        )
        conv3x3 = nn.Sequential(
            nn.BatchNorm2d(num_features=self.output_dim // self.inner_scale),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_dim // self.inner_scale, out_channels=self.output_dim // self.inner_scale, kernel_size=3, padding=1, stride=self.stride),
        )
        conv1x1b = nn.Sequential(
            nn.BatchNorm2d(num_features=self.output_dim // self.inner_scale),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.output_dim // self.inner_scale, out_channels=self.output_dim, kernel_size=1),
        )

        self.activation = nn.Identity() #Override existing activation.
        return nn.ModuleList([conv1x1a, conv3x3, conv1x1b])
    
class StandardModule(ResidualModule):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 stride: int = 1,
                 skip_connection: Optional[nn.Module] = None,
                 p_stochastic_depth: float = 0.):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            skip_connection=skip_connection,
            stride=stride,
            p_stochastic_depth=p_stochastic_depth,
        )

    def _make_convlayers(self):
        conv3x3a = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=self.output_dim, kernel_size=3, padding=1, stride=self.stride),
            nn.BatchNorm2d(num_features=self.output_dim),
            nn.ReLU()
        )
        conv3x3b = nn.Sequential(
            nn.Conv2d(in_channels=self.output_dim, out_channels=self.output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=self.output_dim),
            nn.ReLU(),
        )
        return nn.ModuleList([conv3x3a, conv3x3b])   

def residual_block_factory(block_type: str, **kwargs) -> nn.Module:
    residual_modules = {
        'bottleneck' : BottleneckModule,
        'standard' : StandardModule,
        'bottleneck_pre' : PreActivationBottleneckModule,
    }

    if block_type not in residual_modules:
        raise ValueError(f'[ResidualBlockFactory] Invalid block_type: {block_type}. Choose from {list(residual_modules.keys())}.')
    
    return residual_modules[block_type](**kwargs)

class ResNet(nn.Module):
    def __init__(self, 
                channel_dims: List[int], 
                layer_numbers: List[int], 
                block_types: List[str],
                n_classes: int,
                p_stochastic_depth: float = 0.,
                stochastic_depth_linear_decay: bool = False) -> None:
        super().__init__()
        
        self.layer_numbers = layer_numbers
        self.channel_dims = channel_dims
        self.n_classes = n_classes
        self.block_types = block_types

        if stochastic_depth_linear_decay:
            total_layers = sum(layer_numbers)
            self.kill_probabilities = [i * p_stochastic_depth / total_layers for i in range(total_layers)]
        else:
            self.kill_probabilities = [p_stochastic_depth] * sum(layer_numbers)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channel_dims[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=channel_dims[0]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.convolutional_blocks = nn.ModuleList([
            self._make_stage(
                input_dim=self.channel_dims[max(i-1, 0)],
                stage_dim=self.channel_dims[i],
                num_layers=self.layer_numbers[i],
                downsample=False if i ==0 else True,
                block_type=self.block_types[i],
                kill_probabilities=self.kill_probabilities[sum(self.layer_numbers[:i+1]) - self.layer_numbers[i]:sum(self.layer_numbers[:i+1])]
            ) for i in range(len(self.layer_numbers))
        ])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('n c h w -> n (c h w)'),
            nn.Linear(in_features=self.channel_dims[-1], out_features=n_classes)
        )

        self._initialize_weights()

        print(f'[ResNet] Model successfully initialized.')

    def _initialize_weights(self):
        '''Initialize weights for stem and classifier layers'''
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self,
                    input_dim: int, 
                    stage_dim: int, 
                    num_layers: int, 
                    downsample: bool,
                    block_type: str,
                    kill_probabilities: List[float]) -> nn.Module:
        if not downsample:
            skip_connection = None
        elif block_type != 'bottleneck_pre':
            skip_connection = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=stage_dim, kernel_size=1, stride=2),
                nn.BatchNorm2d(num_features=stage_dim),
            )
        else:
            skip_connection = nn.Sequential(
                nn.BatchNorm2d(num_features=input_dim),
                nn.Conv2d(in_channels=input_dim, out_channels=stage_dim, kernel_size=1, stride=2),
            )

        dsl = residual_block_factory(
            block_type=block_type,
            input_dim=input_dim,
            output_dim=stage_dim,
            stride=2 if downsample else 1,
            skip_connection=skip_connection,
            p_stochastic_depth=kill_probabilities[0]
        )
        
        resnet_blocks = [
            residual_block_factory(
                block_type=block_type,
                input_dim=stage_dim,
                output_dim=stage_dim,
                p_stochastic_depth=kill_probabilities[i]
            )
            for i in range(1, num_layers)
        ]

        return nn.Sequential(dsl, *resnet_blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for convolutional_block in self.convolutional_blocks:
            x = convolutional_block(x)

        return self.classifier(x)
    
    def calculate_flops(self):
        self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)
        print(f'[ResNet] Parameters: {self.params / 1e6 : .2f}M, MACs: {self.macs / 1e9 : .4f}GMACs') #type: ignore

if __name__ == '__main__':
    resnet18 = ResNet(
        channel_dims=[64, 128, 256, 512],
        layer_numbers=[2, 2, 2, 2],
        block_types=['standard']*4,
        n_classes=1000,
        p_stochastic_depth=0.5,
        stochastic_depth_linear_decay=True
    )

    resnet18.calculate_flops()

    resnet34 = ResNet(
        channel_dims=[64, 128, 256, 512],
        layer_numbers=[3, 4, 6, 3],
        block_types=['standard']*4,
        n_classes=1000,
        p_stochastic_depth=0.5,
        stochastic_depth_linear_decay=True
    )

    resnet34.calculate_flops()

    resnet50 = ResNet(
        channel_dims=[256, 512, 1024, 2048],
        layer_numbers=[3, 4, 6, 3],
        block_types=['bottleneck']*4,
        n_classes=1000,
        p_stochastic_depth=0.5,
        stochastic_depth_linear_decay=True
    )

    resnet50.calculate_flops()