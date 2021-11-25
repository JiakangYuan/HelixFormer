import torch
import torch.nn as nn
import models
from .models import register


# return key, query, value and feature map
class LinearCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, kqv_bias=False, norm_layer=nn.LayerNorm, need_reshape=True):
        super(LinearCrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.need_reshape = need_reshape
        self.head_dim = dim // num_heads
        self.kqv = nn.Linear(dim, 3*dim, bias=kqv_bias)
        self.norm = norm_layer(dim)
    
    def forward(self, x):
        if self.need_reshape:
            B, C, H, W = x.shape
            x = x.view(B, C, H*W)
            N = H * W
        else:
            B, C, N = x.shape
        embedding = x.transpose(2, 1).contiguous()
        x = self.norm(embedding)
        kqv = self.kqv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, q, v = kqv[0], kqv[1], kqv[2]
        return k, q, v, embedding


class Mlp(nn.Module):
    def __init__(self, in_feature, hidden_feature=None,
                 out_feature=None, act_layer=nn.GELU):
        super(Mlp, self).__init__()
        out_features = out_feature or in_feature
        self.fc1 = nn.Linear(in_feature, hidden_feature)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LinearHelixTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, kqv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=4):
        super(LinearHelixTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kqv_bias = kqv_bias
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.act = act_layer
        self.atten1 = LinearCrossAttention(dim, num_heads, kqv_bias, norm_layer)
        self.atten2 = LinearCrossAttention(dim, num_heads, kqv_bias, norm_layer)

        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_feature=dim, hidden_feature=mlp_hidden_dim, act_layer=act_layer)
        self.mlp2 = Mlp(in_feature=dim, hidden_feature=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape

        k1, q1, v1, embedding1 = self.atten1(x1)
        k2, q2, v2, embedding2 = self.atten2(x2)
        atten1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        atten2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        atten1 = atten1.softmax(dim=-1)
        atten2 = atten2.softmax(dim=-1)
        N = H *W
        atten1 = (atten1 @ v2).transpose(1,2).reshape(B, N, C)
        atten2 = (atten2 @ v1).transpose(1,2).reshape(B, N, C)

        out1 = self.proj1(atten1)
        out1 = embedding1 + out1
        out1 = out1 + self.mlp1(self.norm1(out1))

        out2 = self.proj2(atten2)
        out2 = embedding2 + out2
        out2 = out2 + self.mlp2(self.norm2(out2))
        
        out1 = out1.transpose(-2, -1).contiguous().view(B, C, H, W)
        out2 = out2.transpose(-2, -1).contiguous().view(B, C, H, W)
        return out1, out2


@register('linear_helix_transformer')
class LinearHelixTransformer(nn.Module):
    def __init__(self, dim, num_heads, kqv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_ratio=4):
        super(LinearHelixTransformer, self).__init__()
        self.layer1 = LinearHelixTransformerBlock(dim, num_heads)

    def forward(self, x1, x2):
        x1, x2 = self.layer1(x1, x2)
        return x1, x2



