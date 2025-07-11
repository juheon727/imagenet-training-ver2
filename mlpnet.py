#Implementation of MLPNet.

import torch
import torch.nn as nn
from calflops import calculate_flops

class Block(nn.Module):
  def __init__(self, input_dim: int, inner_dim, output_dim: int) -> None:
    super().__init__()

    self.ffn = nn.Sequential(
      nn.Linear(input_dim, inner_dim),
      nn.ReLU(),
      nn.Linear(inner_dim, inner_dim),
      nn.ReLU(),
      nn.Linear(inner_dim, output_dim),
      nn.ReLU(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.ffn(x)

class MLPNet136(nn.Module):
  def __init__(self):
    super().__init__()

    self.flatten = nn.Flatten()

    self.ffn0 = Block(3*224*224, 16384, 4096)
    self.stage1 = nn.Sequential(
      *[Block(4096, 16384, 4096) for _ in range(3)]
    )
    self.ffn1 = Block(4096, 8192, 2048)
    self.stage2 = nn.Sequential(
      *[Block(2048, 8192, 2048) for _ in range(12)]
    )
    self.ffn2 = Block(2048, 4096, 1024)
    self.stage3 = nn.Sequential(
      *[Block(1024, 4096, 1024) for _ in range(20)]
    )
    self.ffn3 = Block(1024, 2048, 512)
    self.stage4 = nn.Sequential(
      *[Block(512, 2048, 512) for _ in range(6)]
    )

    self.cls = nn.Linear(512, 1000)

    self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)

    print('[MLPNet-136] Model successfully initialized.')
    print(f'[MLPNet-136] Params: {self.params / 1e6 :.2f}M, MACs: {self.macs / 1e9 :.4f}GMACs') #type: ignore

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.flatten(x)
    x = self.ffn0(x)
    x = self.stage1(x)
    x = self.ffn1(x)
    x = self.stage2(x)
    x = self.ffn2(x)
    x = self.stage3(x)
    x = self.ffn3(x)
    x = self.stage4(x)
    return self.cls(x)
  
class MLPNet46(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.ffn0 = Block(3*224*224, 4096, 2048)
        self.stage1 = nn.Sequential(
            *[Block(2048, 4096, 2048) for _ in range(3)]
        )
        self.ffn1 = Block(2048, 4096, 1024)
        self.stage2 = nn.Sequential(
            *[Block(1024, 2048, 1024) for _ in range(3)]
        )
        self.ffn2 = Block(1024, 2048, 512)
        self.stage3 = nn.Sequential(
            *[Block(512, 1024, 512) for _ in range(6)]
        )
        self.cls = nn.Linear(512, 1000)

        self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)

        print('[MLPNet-46] Model successfully initialized.')
        print(f'[MLPNet-46] Params: {self.params / 1e6 :.2f}M, MACs: {self.macs / 1e9 :.4f}GMACs') #type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.ffn0(x)
        x = self.stage1(x)
        x = self.ffn1(x)
        x = self.stage2(x)
        x = self.ffn2(x)
        x = self.stage3(x)
        return self.cls(x)
  
class MLPNet25(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.ffn0 = Block(3*224*224, 1024, 512)
        self.stage1 = nn.Sequential(
            *[Block(512, 1024, 512) for _ in range(3)]
        )
        self.ffn1 = Block(512, 512, 256)
        self.stage2 = nn.Sequential(
            *[Block(256, 512, 256) for _ in range(3)]
        )
        self.cls = nn.Linear(256, 1000)

        self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)

        print('[MLPNet-25] Model successfully initialized.')
        print(f'[MLPNet-25] Params: {self.params / 1e6 :.2f}M, MACs: {self.macs / 1e9 :.4f}GMACs') #type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.ffn0(x)
        x = self.stage1(x)
        x = self.ffn1(x)
        x = self.stage2(x)
        return self.cls(x)
    
class MLPNet13(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.ffn0 = Block(3*224*224, 256, 256)
        self.stage1 = nn.Sequential(
            *[Block(256, 512, 256) for _ in range(1)]
        )
        self.ffn1 = Block(256, 256, 128)
        self.stage2 = nn.Sequential(
            *[Block(128, 256, 128) for _ in range(1)]
        )
        self.cls = nn.Linear(128, 1000)

        self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)

        print('[MLPNet-13] Model successfully initialized.')
        print(f'[MLPNet-13] Params: {self.params / 1e6 :.2f}M, MACs: {self.macs / 1e9 :.4f}GMACs') #type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.ffn0(x)
        x = self.stage1(x)
        x = self.ffn1(x)
        x = self.stage2(x)
        return self.cls(x)
  
if __name__ == '__main__':
  model = MLPNet13().cuda()
