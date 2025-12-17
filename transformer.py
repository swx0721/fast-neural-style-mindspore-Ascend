# transformer.py - 最终修复版 (解决 MindSpore Pad 维度限制，仅对 H/W 维度填充)
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
# 导入 functional pad (小写 p)
from mindspore.ops import pad as F_pad 

class ReflectionPad(nn.Cell):
    """自定义反射填充层 (仅对 H/W 维度进行填充)"""
    def __init__(self, padding):
        super(ReflectionPad, self).__init__()
        
        if not isinstance(padding, int):
            raise TypeError("ReflectionPad only accepts an integer padding size.")

        # ❗ 关键修复：构造长度为 4 的填充元组，只针对 H 和 W 维度 (最内层的两个维度)
        # 填充顺序：(W_l, W_r, H_t, H_b)
        # MindSpore 的 ops.pad 会从 Tensor 的最内层维度开始应用这些填充值。
        self.paddings_list = [
            padding, padding,  # W 维度填充 (W_l, W_r)
            padding, padding   # H 维度填充 (H_t, H_b)
        ] # 长度为 4，满足平台要求
        self.paddings_tuple = tuple(self.paddings_list)

    def construct(self, x):
        # x 是 [B, C, H, W] 4维。
        # 当 paddings_tuple 长度为 4 时，MindSpore 将其应用于最内层的 H 和 W 维度，
        # 从而避免了 PadV3 对 5 维输入报错的问题。
        return F_pad(x, self.paddings_tuple, "reflect")

class ConvLayer(nn.Cell):
    """反射填充 + 卷积 + GroupNorm + ReLU/无激活"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, relu=True):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        
        self.pad = ReflectionPad(padding)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0, 
            pad_mode='valid' 
        )
        self.in_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=1e-5)    
        self.relu = nn.ReLU() if relu else None

    def construct(self, x):
        x = self.pad(x) 
        x = self.conv(x)
        x = self.in_norm(x)
        if self.relu:
            x = self.relu(x)
        return x

class UpsampleConvLayer(nn.Cell):
    """插值上采样 + 卷积层 (使用反射填充)"""
    def __init__(self, in_channels, out_channels, kernel_size, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        
        self.upsample_factor = upsample
        padding = kernel_size // 2
        
        self.pad = ReflectionPad(padding) 
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0, 
            pad_mode='valid' 
        )
        self.in_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, eps=1e-5)
        self.relu = nn.ReLU()
        self.resize_bilinear = ops.ResizeBilinearV2()

    def construct(self, x):
        if self.upsample_factor:
            _, _, h, w = x.shape
            new_h = h * self.upsample_factor
            new_w = w * self.upsample_factor
            
            x = self.resize_bilinear(x, (new_h, new_w))

        x = self.pad(x)
        x = self.conv(x)
        x = self.in_norm(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Cell):
    """残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, relu=False)

    def construct(self, x):
        return x + self.conv2(self.conv1(x))

class TransformerNet(nn.Cell):
    """适配高分辨率的风格迁移网络"""
    def __init__(self, high_res_mode=True):
        super(TransformerNet, self).__init__()
        self.high_res_mode = high_res_mode
        
        # 下采样
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        if high_res_mode:
            self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=1)
            self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=1)    
        else:
            self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
            self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        
        # 残差块
        res_blocks = 8 if high_res_mode else 5
        self.res_blocks = nn.SequentialCell(
            *[ResidualBlock(128) for _ in range(res_blocks)]
        )
        
        # 上采样
        if high_res_mode:
            self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, upsample=1)
            self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, upsample=1)
        else:
            # standard mode 进行两次上采样 (upsample=2)
            self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, upsample=2)
            self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, upsample=2)
        
        # ❗ 输出层 padding=4
        self.pad_out = ReflectionPad(4)
        self.conv_out = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=0, pad_mode='valid')
        self.tanh = nn.Tanh()

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        
        x = self.pad_out(x)
        x = self.conv_out(x)
        x = self.tanh(x)
        return x