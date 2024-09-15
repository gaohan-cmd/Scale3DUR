import torch
from einops.einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


if __name__ == '__main__':
    # print(to_3d(torch.randn(2, 3, 4, 5)).shape)
    print(torch.randn(2, 60, 3))
    print(to_4d(torch.randn(2, 60, 3), 12, 5).shape)
