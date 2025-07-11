import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x

#x = torch.ones((4, 1, 32, 32), device='cuda', dtype=torch.float32)
x = torch.ones((4, 1, 32, 32), device='cpu', dtype=torch.float32)

#network = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, device='cuda')
network = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, device='cpu')
#apool = nn.AdaptiveAvgPool2d((1, 1)).cuda()
apool = nn.AdaptiveAvgPool2d((1, 1)).cpu()

residual = x
x = network(x)

print(apool(x))
print(apool(residual))

c = torch.ones((2, 2))
b = c
c += 1
print(b)
print(c)

t = torch.ones((4, 1, 32, 32), device='cuda')

tt = t[1:3]
tt = tt + 1

perm = torch.randperm(4)
ttt = t[perm]
ttt += 1

print(apool(t))

identity_blk = nn.Identity()
tttt = identity_blk(t)
tttt += 1

print(apool(t))