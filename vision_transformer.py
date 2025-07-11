import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import transformer_engine.pytorch as te
import einops
from einops.layers.torch import Rearrange
from calflops import calculate_flops

torch.backends.cudnn.benchmark = True

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return einops.rearrange(x, 'b c h w -> b (h w) c')

class VisionTransformer(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, num_heads: int, num_layers: int, patch_size: int, num_classes: int, ffn_multiplier: int, seq_len: int, batch_size: int) -> None:
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.positional_encoding = nn.Parameter(torch.zeros((1, seq_len, embed_dim), dtype=torch.float32))
        self.transformer_encoders = nn.ModuleList([
            te.TransformerLayer(hidden_size=embed_dim, 
                                ffn_hidden_size=embed_dim * ffn_multiplier, 
                                num_attention_heads=num_heads,
                                num_gqa_groups=num_heads,
                                self_attn_mask_type='no_mask',
                                fuse_qkv_params=True,
                                params_dtype=torch.float32,
                                attn_input_format='bshd',
                                hidden_dropout=0.1,
                                attention_dropout=0.1,
                                normalization='LayerNorm',
                                seq_length=seq_len,
                                micro_batch_size=batch_size) for _ in range(num_layers)
        ])
        
        self.ff = nn.Linear(embed_dim, num_classes, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = x + self.positional_encoding
        for layer in self.transformer_encoders:
            x = layer(x)
        x = x.mean(dim=1)
        return self.ff(x)
    
    def calculate_flops(self) -> None:
        self.flops, self.macs, self.params = calculate_flops(model=self, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=4, print_results=False)

        print(f'[VisionTransformer] Parameters: {self.params / 1e6 : .2f}M, MACs: {self.macs / 1e9 : .4f}GMACs') #type: ignore
    
if __name__ == '__main__':
    vit = VisionTransformer(in_channels=3, embed_dim=1280, num_heads=16, num_layers=32, patch_size=14, num_classes=1000, ffn_multiplier=4, seq_len=256, batch_size=1).cuda()
    vit.compile()
    x = torch.ones((1, 3, 224, 224)).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with te.fp8_autocast(enabled=True):
            y = vit(x)
    print(y.shape)
    print(sum(p.numel() for p in vit.parameters()))