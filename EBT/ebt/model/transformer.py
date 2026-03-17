import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and project to embed_dim.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block.
    """

    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class EnergyBasedTransformer(nn.Module):
    """
    Full Energy-Based Transformer.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=37,
        drop=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.energy_head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        h = self.norm(x[:, 0])
        logits = self.energy_head(h)
        return -logits

    def get_features(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        h = self.norm(x[:, 0])
        return h

    def get_attention_maps(self, x):
        attn_maps = []
        x = self.patch_embed(x)
        for block in self.blocks:
            x_norm = block.norm1(x)
            attn_out, attn_weights = block.attn(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False
            )
            attn_maps.append(attn_weights.detach())
            x = x + attn_out
            x = x + block.mlp(block.norm2(x))
        return attn_maps


class PretrainedEBT(nn.Module):
    """
    EBT using a pretrained Vision Transformer (ViT-B/16) as the backbone.
    """

    def __init__(self, num_classes=37):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.vit = vit_b_16(weights=weights)

        embed_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()

        self.num_classes = num_classes
        self.energy_head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.energy_head.weight, std=0.02)
        nn.init.zeros_(self.energy_head.bias)

    def forward(self, x):
        h = self.vit(x)
        logits = self.energy_head(h)
        return -logits

    def get_features(self, x):
        return self.vit(x)

    def get_attention_maps(self, x):
        attn_maps = []
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder.pos_embedding + x
        x = self.vit.encoder.dropout(x)

        for layer in self.vit.encoder.layers:
            x_norm = layer.ln_1(x)
            _, weights = layer.self_attention(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False
            )
            attn_maps.append(weights.detach())
            x = layer(x)

        return attn_maps


def get_model(num_classes=37, img_size=224, size="small", pretrained=False):
    """
    Factory function for EBT models.
    """
    if pretrained:
        model = PretrainedEBT(num_classes=num_classes)
        print("EBT-pretrained: Using ViT-Base backbone (86M params)")
        return model

    configs = {
        "tiny": dict(embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0, drop=0.1),
        "small": dict(embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, drop=0.1),
        "base": dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, drop=0.1),
    }
    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(configs)}")
    cfg = configs[size]
    model = EnergyBasedTransformer(
        img_size=img_size,
        num_classes=num_classes,
        **cfg,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EBT-{size}: {n_params / 1e6:.1f}M parameters, "
          f"depth={cfg['depth']}, embed_dim={cfg['embed_dim']}, heads={cfg['num_heads']}")
    return model
