import torch
import torch.nn as nn

a = torch.randn((256, 3, 32, 32), requires_grad=True, dtype=torch.float16, device='cuda')
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, device='cuda').half()

perm = torch.randperm(256)
shuffled = a[perm]
shuffled = conv(shuffled)

shuffled = shuffled.mean()

shuffled.backward()

x = torch.ones((3, 3), device='cuda', requires_grad=True)
y = x ** 2

y[0] = 0
y.mean().backward()