import torch.nn as nn
from typing import Optional
import torch


__all__ = ['C3k2B']
 
# class AConv(nn.Module):
#     """AConv."""

#     def __init__(self, c1, c2):
#         """Initializes AConv module with convolution layers."""
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 3, 2, 1)

#     def forward(self, x):
#         """Forward pass through AConv layer."""
#         x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         return self.cv1(x) 
# class SCDown(nn.Module):
#     """
#     SCDown module for downsampling with separable convolutions.

#     This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
#     efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

#     Attributes:
#         cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
#         cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

#     Methods:
#         forward: Applies the SCDown module to the input tensor.

#     Examples:
#         >>> import torch
#         >>> from ultralytics import SCDown
#         >>> model = SCDown(c1=64, c2=128, k=3, s=2)
#         >>> x = torch.randn(1, 64, 128, 128)
#         >>> y = model(x)
#         >>> print(y.shape)
#         torch.Size([1, 128, 64, 64])
#     """

#     def __init__(self, c1, c2, k, s):
#         """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

#     def forward(self, x):
#         """Applies convolution and downsampling to the input tensor in the SCDown module."""
#         return self.cv2(self.cv1(x))
 
# def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
#     conv = nn.Sequential()
#     padding = (kernel_size - 1) // 2
#     conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
#     if norm:
#         conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
#     if act:
#         conv.add_module('Activation', nn.ReLU6())
#     return conv



                    
# class Bottleneck(nn.Module):
#     """Standard bottleneck."""

#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         """Applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x)) 

#VoVNet
# class Bottleneck(nn.Module):
#     """Custom bottleneck module that splits input into two paths and concatenates the outputs."""
#     def __init__(self, c1, c2, g=1, k=(3, 3), e=0.5):
#         super().__init__()
#         c_ = int(c2 * e) # hidden channels
#         self.cv1 = Conv(c1, c_, 3, 1) # Adjust input channels to half
#         self.cv2 = Conv(c_, c_, k[1], 1)
#         self.cv3 = Conv(c_, c2, k[0], 1) # Adjust input channels to half for the second path
#     def forward(self, x):
#         out1 = self.cv1(x)
#         out2 = self.cv2(out1)
#         out3 = self.cv3(out2)
#         out = x + out1 + out2 + out3
#         return out # 通过最后一个卷积层
# 2B
class Bottleneck(nn.Module):
    """Custom bottleneck module that splits input into two paths and concatenates the outputs."""
 
    def __init__(self, c1, c2, g=1, k=(3, 3), e=0.5):
        """Initializes the custom bottleneck module."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * c_, 1, 1)  # Adjust input channels to half
        self.cv2 = Conv(c_, c_, k[1], 1, g=g)
        self.cv3 = Conv(c_, c_, k[0], 1)  # Adjust input channels to half for the second path
        self.cv4 = Conv(2 * c_, c2, 1, 1)
 
    def forward(self, x):
        
        x1, x2 = self.cv1(x).chunk(2, dim=1)  # 在通道维度上将输入分割成两部分
        x3 = self.cv2(x1)  # 对第一部分应用卷积
        x4 = self.cv3(x2)  # 对第二部分应用卷积
        # 将x3和x4沿着通道维度（dim=1）拼接起来
        x_concatenated = torch.cat((x3, x4), dim=1)
        return self.cv4(x_concatenated)  # 通过最后一个卷积层

# class Bottleneck(nn.Module):
#     """Custom bottleneck module that splits input into two paths and concatenates the outputs."""
 
#     def __init__(self, c1, c2, g=1, k=(3, 3), e=0.5):
#         """Initializes the custom bottleneck module."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * c_, 1, 1)  # Adjust input channels to half
#         self.cv2 = SCDown(c_, c_, k[1], 1)
#         self.cv3 = SCDown(c_, c_, k[0], 1)  # Adjust input channels to half for the second path
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.cv5 = Conv(c_, c_, 3, 1)
#         self.cv6 = Conv(c_, c_, 3, 1)
 
#     def forward(self, x):
        
#         x1, x2 = self.cv1(x).chunk(2, dim=1)  # 在通道维度上将输入分割成两部分
#         x3 = self.cv2(self.cv5(x1))  # 对第一部分应用卷积
#         x4 = self.cv3(self.cv6(x2))  # 对第二部分应用卷积
#         # 将x3和x4沿着通道维度（dim=1）拼接起来
#         x_concatenated = torch.cat((x3, x4), dim=1)
#         return self.cv4(x_concatenated)  # 通过最后一个卷积层
 
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, g, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k2B(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, g, e=1.0) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, g, e=1.0) for _ in range(n)))
 
 

 
 
if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)
    model = C3k2B(32, 32)
    print(model(x).shape)