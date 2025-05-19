from collections import OrderedDict

import torch
from torch import nn
from functools import partial


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def drop_path(x, drop_prob, training: bool = False):
    if drop_prob == 0 or not training:
        return x
    return x


class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layers=None):
        super(PatchEmbedding, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.norm_layers = norm_layers
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.conv = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layers if self.norm_layers else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        # [B,C,H,W]-->[B,C,HW]
        # [B,C,HW]-->[B,HW,C]  [B,16*16,768]
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_features, hidden_features=None, output_features=None, drop=0.):
        super(MLP, self).__init__()
        self.output_features = output_features or input_features
        self.hidden_features = hidden_features or input_features
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, atten_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.qk_scale = qk_scale or head_dim ** (-0.5)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.atten_drop = nn.Dropout(atten_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv->[batch_size, num_patches + 1, 3*total_embed_dim]
        # reshape->[batch_size, num_patches + 1, 3,num_heads,head_dim]
        # permute->[3,batch_size,num_heads,num_patches + 1,head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v ->[batch_size,num_heads,num_patches + 1,head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose->[batch_size,num_heads,head_dim,num_patches + 1]
        # qk->[batch_size,num_heads,num_patches + 1,num_patches + 1]
        qk = q @ k.transpose(-2, -1) * self.qk_scale
        qk = qk.softmax(dim=-1)
        qk = self.atten_drop(qk)

        # qkv->[batch_size,num_heads,num_patches + 1,head_dim]
        qkv = qk @ v
        # transpose->[batch_size,num_patches + 1,num_heads,head_dim]
        # reshape->[batch_size,num_patches + 1,total_embed_dim]
        qkv = qkv.transpose(1, 2).reshape(B, N, C)
        qkv = self.proj_drop(self.proj(qkv))
        return qkv


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_ratio=4, qkv_bias=False, qk_scale=None, drop_path_ratio=0.,
                 atten_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Block, self).__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.multi_atten = Attention(dim, num_heads, qkv_bias, qk_scale, atten_drop_ratio, proj_drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0 else nn.Identity()
        mlp_hidden_dim = int(mlp_hidden_ratio * dim)
        self.mlp = MLP(input_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.dropout(self.multi_atten(self.layernorm(x)))
        x = x + self.dropout(self.mlp(self.layernorm(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_c=3,
                 embed_dim=768,
                 encoder_layer_num=12,
                 num_classes=1000,
                 num_heads=12,
                 mlp_hidden_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 representation_size=None,
                 distilled=False,
                 drop_ratio=0.,
                 drop_path_ratio=0.,
                 atten_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 norm_layer=None,
                ):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_token = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbedding(img_size=image_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, encoder_layer_num)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_hidden_ratio=mlp_hidden_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop_path_ratio=drop_path_ratio,
                  atten_drop_ratio=atten_drop_ratio, proj_drop_ratio=proj_drop_ratio)
            for i in range(encoder_layer_num)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_feature(self, x):
        # x->[B,N,H,W]
        B, N, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
