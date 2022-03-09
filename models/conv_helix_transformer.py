from einops import rearrange
import torch
from torch import nn, einsum
from .models import register


# layer norm, but done in the channel dimension
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super(DepthWiseConv2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                      groups=dim_in, stride=stride, bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.fn = fn

    def forward(self, x1, x2, **kwargs):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        return self.fn(x1, x2, **kwargs)


class UnidirectionalPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(UnidirectionalPreNorm, self).__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super(FeedForward, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        return self.net1(x1), self.net2(x2)


class UnidirectionalFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super(UnidirectionalFeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ConvCrossAttention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=512, dropout=0.):
        super(ConvCrossAttention, self).__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q1 = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_kv1 = DepthWiseConv2d(dim, inner_dim*2, proj_kernel, padding=padding, stride=kv_proj_stride, bias=False)
        
        self.to_q2 = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_kv2 = DepthWiseConv2d(dim, inner_dim*2, proj_kernel, padding=padding, stride=kv_proj_stride, bias=False)
        
        self.to_out1 = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        self.to_out2 = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        shape = x1.shape
        b, n, _, y, h = *shape, self.heads

        q1, k1, v1 = (self.to_q1(x1), *self.to_kv1(x1).chunk(2, dim = 1))
        q2, k2, v2 = (self.to_q2(x2), *self.to_kv2(x2).chunk(2, dim = 1))

        q1, k1, v1 = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q1, k1, v1))
        q2, k2, v2 = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q2, k2, v2))
        
        dots1 = einsum('b i d, b j d -> b i j', q1, k2) * self.scale
        dots2 = einsum('b i d, b j d -> b i j', q2, k1) * self.scale
        
        attn1 = self.attend(dots1)
        attn2 = self.attend(dots2)

        out1 = einsum('b i j, b j d -> b i d', attn1, v2)
        out2 = einsum('b i j, b j d -> b i d', attn2, v1)
        
        out1 = rearrange(out1, '(b h) (x y) d -> b (h d) x y', h=h, y=y).contiguous()
        out2 = rearrange(out2, '(b h) (x y) d -> b (h d) x y', h=h, y=y).contiguous()
        
        return self.to_out1(out1), self.to_out2(out2)


class UnidirectionalConvCrossAttention(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, heads=8, dim_head=64, dropout=0.):
        super(UnidirectionalConvCrossAttention, self).__init__()
        inner_dim = dim_head * heads
        padding = proj_kernel // 2
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_kv = DepthWiseConv2d(dim, inner_dim*2, proj_kernel, padding=padding, stride=kv_proj_stride, bias=False)
        self.to_q = DepthWiseConv2d(dim, inner_dim, proj_kernel, padding=padding, stride=1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x1, x2):
        shape = x1.shape
        b, n, _, y, h = *shape, self.heads
        k, v = self.to_kv(x2).chunk(2, dim=1)
        q = self.to_q(x1)
        (q, k, v) = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h).contiguous(), (q, k, v))
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, y=y).contiguous()
        return self.to_out(out)


class ConvCrossTransformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super(ConvCrossTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvCrossAttention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
                                                heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))
        
    def forward(self, x1, x2):
        for attn, ff in self.layers:
            temp1, temp2 = attn(x1, x2)
            x1 = x1 * temp1
            x2 = x2 * temp2
            temp1, temp2 = ff(x1, x2)
            x1 = temp1 + x1
            x2 = temp2 + x2
        return x1, x2


# without feedforward
class TransformerWithoutFFD(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4, dropout=0.):
        super(TransformerWithoutFFD, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvCrossAttention(dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
                                                heads=heads, dim_head=dim_head, dropout=dropout)),
            ]))
        
    def forward(self, x1, x2):
        for attn in self.layers:
            temp1, temp2 = attn(x1, x2)
            x1 = x1 + temp1
            x2 = x2 + temp2
        return x1, x2


class UnidirectionalConvCrossTransformer(nn.Module):
    def __init__(self, dim, proj_kernel, kv_proj_stride, depth, heads, dim_head=64, mlp_mult=4,
                 dropout=0., unidirectional_side='support'):
        super(UnidirectionalConvCrossTransformer, self).__init__()
        self.unidirectional_side = unidirectional_side
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, UnidirectionalConvCrossAttention(dim, proj_kernel=proj_kernel,
                                                              kv_proj_stride=kv_proj_stride,
                                                              heads=heads, dim_head=dim_head, dropout=dropout)),
                UnidirectionalPreNorm(dim, UnidirectionalFeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x1, x2):
        for attn, ff in self.layers:
            if self.unidirectional_side == 'support':
                temp1 = attn(x1, x2)
                x1 = x1 + temp1
                x1 = ff(x1) + x1
            else:
                temp2 = attn(x2, x1)
                x2 = x2 + temp2
                x2 = ff(x2) + x2
        return x1, x2


# our proposed helix-transformer: symmetric, bidirectional
@register('conv_helix_transformer')
class ConvHelixTransformer(nn.Module):
    def __init__(self, dim=64, emb_dim=64, emb_kernel=3, emb_stride=1, proj_kernel=3, kv_proj_stride=1,
                    heads=2, dim_head=64, depth=1, mlp_mult=4, dropout=0.):
        super(ConvHelixTransformer, self).__init__()
        self.conv1 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.conv2 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.LayerNorm1 = LayerNorm(emb_dim)
        self.LayerNorm2 = LayerNorm(emb_dim)
        self.transformer = ConvCrossTransformer(dim=emb_dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
                                                depth=depth, heads=heads, dim_head=dim_head, mlp_mult=mlp_mult,
                                                dropout=dropout)
        
    def forward(self, x1, x2):
        x1 = self.LayerNorm1(self.conv1(x1))
        x2 = self.conv2(x2)
        x2 = self.LayerNorm2(x2)
        x1, x2 = self.transformer(x1, x2)
        return x1, x2


# unidirectional version in insight analysis
@register('unidirectional_helix_transformer')
class UnidirectionalHelixTransformer(nn.Module):
    def __init__(self, dim=64, emb_dim=64, emb_kernel=3, emb_stride=1, proj_kernel=3, kv_proj_stride=1,
                 heads=2, dim_head=64, depth=1, mlp_mult=4, dropout=0., unidirectional_side='support'):
        super(UnidirectionalHelixTransformer, self).__init__()
        self.conv1 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.conv2 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.LayerNorm1 = LayerNorm(emb_dim)
        self.LayerNorm2 = LayerNorm(emb_dim)
        self.transformer = UnidirectionalConvCrossTransformer(dim=emb_dim, proj_kernel=proj_kernel,
                                                              kv_proj_stride=kv_proj_stride, depth=depth,
                                                              heads=heads, dim_head=dim_head, mlp_mult=mlp_mult,
                                                              dropout=dropout, unidirectional_side=unidirectional_side)
        
    def forward(self, x1, x2):
        x1 = self.LayerNorm1(self.conv1(x1))
        x2 = self.LayerNorm2(self.conv2(x2))
        x1, x2 = self.transformer(x1, x2)
        return x1, x2


# asymmetric version in insight analysis and supplementary material
@register('asymmetric_helix_transformer')
class AsymmetricHelixTransformer(nn.Module):
    def __init__(self, dim=64, emb_dim=64, emb_kernel=3, emb_stride=1, proj_kernel=3, kv_proj_stride=1,
                 heads=2, dim_head=64, depth=1, mlp_mult=4, dropout=0.):
        super(AsymmetricHelixTransformer, self).__init__()
        self.conv1 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.conv2 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.LayerNorm1 = LayerNorm(emb_dim)
        self.LayerNorm2 = LayerNorm(emb_dim)
        self.transformer1 = UnidirectionalConvCrossTransformer(dim=emb_dim, proj_kernel=proj_kernel,
                                                               kv_proj_stride=kv_proj_stride, depth=depth, heads=heads,
                                                               dim_head=dim_head, mlp_mult=mlp_mult, dropout=dropout,
                                                               unidirectional_side='query')
        self.transformer2 = UnidirectionalConvCrossTransformer(dim=emb_dim, proj_kernel=proj_kernel,
                                                               kv_proj_stride=kv_proj_stride, depth=depth, heads=heads,
                                                               dim_head=dim_head, mlp_mult=mlp_mult, dropout=dropout,
                                                               unidirectional_side='support')

    def forward(self, x1, x2):
        x1 = self.LayerNorm1(self.conv1(x1))
        x2 = self.LayerNorm2(self.conv2(x2))
        x1, x2 = self.transformer1(x1, x2)
        x1, x2 = self.transformer2(x1, x2)
        return x1, x2


# without relation fusion in insight analysis
@register('conv_helix_transformer_withoutffd')
class HelixTransformerWithoutFFD(nn.Module):
    def __init__(self, dim=64, emb_dim=64, emb_kernel=3, emb_stride=1, proj_kernel=3, kv_proj_stride=1,
                 heads=2, dim_head=64, depth=1, mlp_mult=4, dropout=0.):
        super(HelixTransformerWithoutFFD, self).__init__()
        self.conv1 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.conv2 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.LayerNorm1 = LayerNorm(emb_dim)
        self.LayerNorm2 = LayerNorm(emb_dim)
        self.transformer = TransformerWithoutFFD(dim=emb_dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
                                                 depth=depth, heads=heads, dim_head=dim_head, mlp_mult=mlp_mult,
                                                 dropout=dropout)
        
    def forward(self, x1, x2):
        x1 = self.LayerNorm1(self.conv1(x1))
        x2 = self.LayerNorm2(self.conv2(x2))
        x1, x2 = self.transformer(x1, x2)
        return x1, x2


# stack helix-transformer in supplementary material
@register('stack_trans_net_cvt')
class StackHelixTransformer(nn.Module):
    def __init__(self, dim=64, emb_dim=64, emb_kernel=3, emb_stride=1, proj_kernel=3, kv_proj_stride=1,
                 heads=2, dim_head=64, depth=1, mlp_mult=4, dropout=0.):
        super(StackHelixTransformer, self).__init__()
        self.conv1 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.conv2 = nn.Conv2d(dim, emb_dim, kernel_size=emb_kernel, padding=emb_kernel // 2, stride=emb_stride)
        self.LayerNorm1 = LayerNorm(emb_dim)
        self.LayerNorm2 = LayerNorm(emb_dim)
        self.transformer1 = ConvCrossTransformer(dim=emb_dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
                                                 depth=depth, heads=heads, dim_head=dim_head, mlp_mult=mlp_mult,
                                                 dropout=dropout)
        self.transformer2 = ConvCrossTransformer(dim=emb_dim, proj_kernel=proj_kernel, kv_proj_stride=kv_proj_stride,
                                                 depth=depth, heads=heads, dim_head=dim_head, mlp_mult=mlp_mult,
                                                 dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.LayerNorm1(self.conv1(x1))
        x2 = self.LayerNorm2(self.conv2(x2))
        x1, x2 = self.transformer1(x1, x2)
        x1, x2 = self.transformer2(x1, x2)
        return x1, x2
