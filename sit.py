import torch
from torch import nn
import numpy as np
from einops import repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SiT(nn.Module):
    def __init__(self, image_size, patch_size, rotation_node, contrastive_head, dim=768, depth=12, heads=12, pool='cls', in_channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., scale_dim=4):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        h = image_size // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, dim))
        self.rotation_token = nn.Parameter(torch.randn(1, 1, dim))
        self.contrastive_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)



        self.pool = pool
        self.to_img = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=h, p1=patch_size, p2=patch_size, c=in_channels)
        )

        self.rot_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, rotation_node)
        )

        self.contrastive_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, contrastive_head)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        rotation_tokens = repeat(self.rotation_token, '() n d -> b n d', b=b)
        contrastive_tokens = repeat(self.contrastive_token, '() n d -> b n d', b=b)
        x = torch.cat((rotation_tokens, contrastive_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 2)]
        x = self.dropout(x)

        x = self.transformer(x)

        l_rotation = x[:, 0]
        l_contrastive = x[:, 1]

        return self.rot_head(l_rotation), self.contrastive_head(l_contrastive), self.to_img(x[:, 2:])


if __name__ == "__main__":
    img = torch.ones([1, 3, 224, 224])

    model = SiT(image_size=224, patch_size=16, rotation_node=4, contrastive_head=512)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    l_rotation, l_contrastive, out_img = model(img)

    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]
    print("Shape of l_rotation :", l_rotation.shape) # [B, rotation_node]
    print("Shape of l_contrastive :", l_contrastive.shape) # [B, contrastive_head]