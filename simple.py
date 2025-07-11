import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
#import transformer_engine.pytorch as te
from calflops import calculate_flops

class SimpleCNN(nn.Module):
    def __init__(self,
                 n_classes: int) -> None:
        """
        A Convolutional Neural Network(CNN) with a very basic architecture.

        Args:
            n_classes (int): Number of classes for classification.
        """
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=4),
            nn.BatchNorm2d(num_features=32),
            nn.SiLU(),
        )

        self.conv1a = self._conv_type1(input_dim=32, output_dim=64)
        self.conv1b1 = self._conv_type2(dim=64, expand_factor=4)
        self.conv1b2 = self._conv_type2(dim=64, expand_factor=4)
        self.se1 = self._se(input_dim=64, contraction_factor=4)

        self.conv2a = self._conv_type1(input_dim=64, output_dim=128)
        self.conv2b1 = self._conv_type2(dim=128, expand_factor=4)
        self.conv2b2 = self._conv_type2(dim=128, expand_factor=4)
        self.se2 = self._se(input_dim=128, contraction_factor=4)

        self.ffn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('n c h w -> n (c h w)'),
            nn.Linear(128, n_classes)
        )

        print('[SimpleCNN] Model successfully initialized.')

    @staticmethod
    def _conv_type1(input_dim: int,
                    output_dim: int) -> nn.Module:
        """
        A Type-1 Convolution block that reduces spatial dimensions with stride=2.

        Args:
            input_dim (int): Number of channels of the input tensor.
            output_dim (int): Number of channels of the output tensor.

        Returns:
            nn.Module: Type-1 Convolution block with specified arguments.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=7, padding=3, groups=input_dim, stride=2),
            nn.BatchNorm2d(num_features=input_dim),
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1),
            nn.BatchNorm2d(num_features=output_dim),
            nn.SiLU(),
        )

    @staticmethod
    def _conv_type2(dim: int, expand_factor: int) -> nn.Module:
        """
        A Type-2 Convolution block.

        Args:
            input_dim (int): Number of channels of the input tensor.
            expand_factor (int): Expansion factor of 1x1 Convolutions.

        Returns:
            nn.Module: Type-2 Convolution block with specified arguments.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim),
            nn.BatchNorm2d(num_features=dim),
            nn.Conv2d(in_channels=dim, out_channels=dim * expand_factor, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=dim * expand_factor, out_channels=dim, kernel_size=1),
            nn.BatchNorm2d(num_features=dim),
        )

    @staticmethod
    def _se(input_dim: int, contraction_factor: int) -> nn.Module:
        """
        A SE(Squeeze-and-Excite) Block.

        Args:
            input_dim (int): Number of channels of the input tensor.
            contraction_factor (int): Contraction factor of 1x1 Convolutions.

        Returns:
            nn.Module: SE block with specified arguments.
        """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('n c h w -> n h w c'),
            nn.Linear(input_dim, input_dim // contraction_factor, bias=False),
            nn.SiLU(),
            nn.Linear(input_dim // contraction_factor, input_dim, bias=False),
            Rearrange('n h w c -> n c h w'),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)

        y = self.conv1a(y)
        y = y + self.conv1b1(y)
        y = y + self.conv1b2(y)
        y = y * self.se1(y)

        y = self.conv2a(y)
        y = y + self.conv2b1(y)
        y = y + self.conv2b2(y)
        y = y * self.se2(y)

        return self.ffn(y)

    def calculate_flops(self) -> None:
        with torch.no_grad():
            self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)
            print(f'[SimpleCNN] Parameters: {self.params / 1e6 : .2f}M, MACs: {self.macs / 1e9 : .4f}GMACs') #type: ignore
            
class SEBlock(nn.Module):
    """
    A SE(Squeeze-and-Excite) Block.

    Args:
        input_dim (int): Number of channels of the input tensor.
        contraction_factor (int): Contraction factor of 1x1 Convolutions.
    """
    def  __init__(self, input_dim: int, contraction_factor: int) -> None:
        super().__init__()

        self.apool = nn.AdaptiveAvgPool2d((1, 1))
        self.se = nn.Sequential(
            Rearrange('n c h w -> n h w c'),
            nn.Linear(input_dim, input_dim // contraction_factor, bias=False),
            nn.SiLU(),
            nn.Linear(input_dim // contraction_factor, input_dim, bias=False),
            Rearrange('n h w c -> n c h w'),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.apool(x)
        return x * self.se(y)

class ChannelShift(nn.Module):
    def __init__(self, shift_size: int, inverse_shift: bool) -> None:
        super().__init__()
        self.shift_size = -shift_size if inverse_shift else shift_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, shifts=self.shift_size, dims=1)
    
class ChannelShuffle(nn.Module):
    def __init__(self, cardinality: int) -> None:
        super().__init__()
        self.cardinality = cardinality
        self.shuffle = Rearrange('n (k c) h w -> n (c k) h w', k=self.cardinality)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(x)
    
class ConvType2(nn.Module):
    def __init__(self, dim: int, expand_factor: int, cardinality: int) -> None:
        super().__init__()

        self.conv2a = nn.Conv2d(in_channels=dim, out_channels=dim * expand_factor, kernel_size=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=dim * expand_factor)
        #self.se_block = SEBlock(input_dim=dim * expand_factor, contraction_factor=expand_factor)
        self.conv1 = nn.Conv2d(in_channels=dim * expand_factor, out_channels=dim * expand_factor, kernel_size=3, padding=1, groups=dim * expand_factor, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=dim * expand_factor)
        self.conv2b = nn.Conv2d(in_channels=dim * expand_factor, out_channels=dim, kernel_size=1, groups=cardinality, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=dim)
        self.se_block = SEBlock(input_dim=dim, contraction_factor=expand_factor * 4)

        #self.activation = nn.SiLU()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = self.conv2a(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        #x = self.se_block(x)
        x = self.conv2b(x)
        x = self.bn3(x)
        x = self.se_block(x)

        return self.activation(residual + x)

class SimpleCNNv2(nn.Module):
    def __init__(self,
                 n_classes: int) -> None:
        """
        A Convolutional Neural Network(CNN) with a very basic architecture.

        Args:
            n_classes (int): Number of classes for classification.
        """
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(num_features=96),
            #nn.SiLU(),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(*[
            nn.Sequential(
                ConvType2(dim=96, expand_factor=4, cardinality=8),
                #ChannelShift(shift_size=12, inverse_shift=((i%2)!=0)),
                ChannelShuffle(cardinality=8),
            ) for i in range(2)
        ])

        self.conv2a = self._conv_type1(input_dim=96, output_dim=192)
        self.conv2b = nn.Sequential(*[
            nn.Sequential(
                ConvType2(dim=192, expand_factor=4, cardinality=8),
                #ChannelShift(shift_size=24, inverse_shift=((i%2)!=0)),
                ChannelShuffle(cardinality=8),
            ) for i in range(2)
        ])

        self.conv3a = self._conv_type1(input_dim=192, output_dim=384)
        self.conv3b = nn.Sequential(*[
            nn.Sequential(
                ConvType2(dim=384, expand_factor=4, cardinality=8),
                #ChannelShift(shift_size=48, inverse_shift=((i%2)!=0)),
                ChannelShuffle(cardinality=8)
            ) for i in range(6)
        ])

        self.conv4a = self._conv_type1(input_dim=384, output_dim=768)
        self.conv4b = nn.Sequential(*[
            nn.Sequential(
                ConvType2(dim=768, expand_factor=4, cardinality=8),
                #ChannelShift(shift_size=96, inverse_shift=((i%2)!=0)),
                ChannelShuffle(cardinality=8)
            ) for i in range(2)
        ])

        self.ffn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('n c h w -> n (c h w)'),
            nn.Linear(768, n_classes)
        )

        print('[SimpleCNNv2] Model successfully initialized.')

    @staticmethod
    def _conv_type1(input_dim: int,
                    output_dim: int) -> nn.Module:
        """
        A Type-1 Convolution block that reduces spatial dimensions with stride=2.

        Args:
            input_dim (int): Number of channels of the input tensor.
            output_dim (int): Number of channels of the output tensor.

        Returns:
            nn.Module: Type-1 Convolution block with specified arguments.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=output_dim),
            #nn.SiLU(),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.conv1(y)
        y = self.conv2a(y)
        y = self.conv2b(y)
        y = self.conv3a(y)
        y = self.conv3b(y)
        y = self.conv4a(y)
        y = self.conv4b(y)
        return self.ffn(y)

    def calculate_flops(self) -> None:
        with torch.no_grad():
            self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)
            print(f'[SimpleCNNv2] Parameters: {self.params / 1e6 : .2f}M, MACs: {self.macs / 1e9 : .4f}GMACs') #type: ignore

if __name__ == '__main__':
    model = SimpleCNN(n_classes=10)
    model.calculate_flops()

    model = SimpleCNNv2(n_classes=1000)
    model.calculate_flops()
