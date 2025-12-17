# vgg.py - 最终修正版（解决了 MindSpore Cast API 错误和 VGG 权重加载问题）
import mindspore.nn as nn
from mindspore import Tensor, ops, dtype as mstype
import numpy as np
from mindcv.models import vgg19
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import os

# 默认的 VGG 权重文件名
VGG_CKPT_FILENAME = 'vgg19-bedee7b6.ckpt'

class VGG19_Feature(nn.Cell):
    """
    VGG19 特征提取器，用于内容 & 风格提取。
    强制从本地路径加载权重，并修正键名以匹配 MindCV 结构。
    """
    def __init__(self, ckpt_path=VGG_CKPT_FILENAME):
        super().__init__()
        
        # 1. 加载模型结构
        self.vgg = vgg19(pretrained=False)
        
        # 2. 显式加载本地权重
        if not os.path.exists(ckpt_path):
             raise FileNotFoundError(f"❌ 错误：VGG 权重文件未找到于路径 '{ckpt_path}'。请确保文件存在。")
             
        param_dict = load_checkpoint(ckpt_path)
        
        # 关键修正：重命名参数以匹配网络中的键名（将 'features.x.weight' -> 'vgg.features.x.weight'）
        new_param_dict = {}
        for name, param in param_dict.items():
            # 只有features和classifier是需要加载的，对所有参数添加 'vgg.' 前缀
            new_param_dict['vgg.' + name] = param
            
        # 加载参数。关于 classifier 的未加载警告是正常的，因为我们不需要分类层。
        load_param_into_net(self.vgg, new_param_dict)
        print(f"✅ VGG19 features loaded successfully from local file: {ckpt_path}")


        self.vgg_layers = self.vgg.features
        for p in self.vgg_layers.get_parameters():
            p.requires_grad = False # 冻结 VGG 权重

        # BGR mean
        mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape(1, 3, 1, 1)
        self.mean_tensor = Tensor(mean_bgr, mstype.float32)

        # 层索引
        self.layer_names = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_4', '26': 'relu4_4', '35': 'relu5_4'}
        self.cat = ops.Concat(1)
        # 修正：ops.Cast 初始化时不接受参数
        self.cast = ops.Cast() 

    def construct(self, x):
        """
        x: RGB [-1,1] tensor (来自 TransformerNet 的输出或 Loss 的输入)
        输出: dict{layer_name: feature tensor}
        """
        
        # 将 [-1, 1] 范围转换到 [0, 255]，以匹配 VGG 预训练的输入规范
        x_scaled = (x + 1.0) * 127.5 
        
        # RGB -> BGR 顺序
        x_bgr = self.cat([x_scaled[:, 2:3, :, :], x_scaled[:, 1:2, :, :], x_scaled[:, 0:1, :, :]])
        
        # 减去 BGR 均值
        out = x_bgr - self.mean_tensor

        features = {}
        for i, layer in enumerate(self.vgg_layers):
            out = layer(out)
            if str(i) in self.layer_names:
                # 修正：在 construct 中传入目标数据类型
                features[self.layer_names[str(i)]] = self.cast(out, mstype.float32)
            if i == 35:
                break
        return features