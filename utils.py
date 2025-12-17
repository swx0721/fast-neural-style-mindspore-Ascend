# utils.py
import numpy as np
import cv2
from mindspore import Tensor, ops, dtype as mstype
import matplotlib.pyplot as plt
# ------------------ 1. 全局配置 ------------------
# 填充宽度：输入时使用
PAD = 16 
# 裁剪边距：主动丢弃边缘，确保输出稳定
# 修复：增大裁剪边距以移除反射填充导致的镜像伪影
TOTAL_CROP_MARGIN = 0 # <--- 关键修复：从 20 提高到 50
# 最大边长限制：推理时先缩小的目标尺寸
MAX_SIDE_SIZE = 720 

# ------------------ Gram Matrix/加载/保存/plot_losses 保持不变 ------------------

def gram(tensor):
    B, C, H, W = tensor.shape
    x = ops.reshape(tensor, (B, C, H*W))
    x_t = ops.transpose(x, (0, 2, 1))
    # MindSpore 默认 Tensor 乘法是矩阵乘法
    return ops.matmul(x, x_t) / (C * H * W)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Failed to load image {path}")
        return None
    return img

def saveimg(img, path):
    cv2.imwrite(path, np.clip(img, 0, 255).astype(np.uint8))

def plot_losses(total, content, style, tv, save_path="loss_curve.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(total, label='Total Loss', color='blue')
    plt.plot(content, label='Content Loss', color='orange', alpha=0.8)
    plt.plot(style, label='Style Loss', color='green', alpha=0.8)
    plt.plot(tv, label='TV Loss', color='red', alpha=0.8)
    plt.xlabel('Steps')
    plt.ylabel('Loss Value')
    plt.title('Training Process Losses')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    # 如果 TV Loss 数值特别大，可以考虑开启对数坐标轴
    # plt.yscale('log') 
    plt.savefig(save_path)
    plt.close() # 释放内存

# ------------------ 2. 图像 <-> MindSpore Tensor (带缩放和填充) ------------------

def itot(img, max_size=None):
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # 1. 缩放逻辑：将最长边限制在 max_size (即 MAX_SIDE_SIZE)
    if max_size is not None:
        max_side = max(h, w)
        if max_side > max_size:
            scale = max_size / float(max_side)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # 使用 INTER_AREA 进行高质量下采样
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 2. 边缘填充：使用镜像填充 (BORDER_REFLECT_101)
    img = cv2.copyMakeBorder(
        img, 
        PAD, PAD, PAD, PAD, 
        cv2.BORDER_CONSTANT, value=0 # 确保是零填充 
    )
    
    # 3. 归一化和 CHW 转换
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_chw = img_rgb.transpose(2, 0, 1)
    # Tensor 范围 [-1, 1] 转换
    img_chw = img_chw * 2.0 - 1.0 
    tensor = Tensor(np.expand_dims(img_chw, 0), dtype=mstype.float32)
    return tensor

def ttoi(tensor):
    """
    将 TransformerNet 输出的 Tensor (B, C, H, W) 范围 [-1, 1] 
    正确反归一化为图像 (H, W, C) 范围 [0, 255]
    """
    # 1. Tensor -> NumPy array，并取第一个 batch
    img = tensor.asnumpy()[0] 
    
    # 2. 关键修正：从 [-1, 1] 映射到 [0, 255]
    # 公式：(x + 1) / 2 * 255.0 = x * 127.5 + 127.5
    img = (img + 1.0) * 127.5
    
    # 3. 转换通道顺序 (C, H, W -> H, W, C)
    img = np.transpose(img, (1, 2, 0)) 
    
    # 4. 转换颜色空间 (RGB -> BGR) 并裁剪到 [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # TransformerNet 默认是 RGB 输出
    return img

# ------------------ 3. 色彩迁移 (保持不变) ------------------
def transfer_color(src, dest):
    try:
        # 确保输入图像的形状一致
        if src.shape[:2] != dest.shape[:2]:
             src = cv2.resize(src, (dest.shape[1], dest.shape[0]), interpolation=cv2.INTER_LINEAR)

        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        dest_lab = cv2.cvtColor(dest, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 风格化输出占 80%，原图占 20%
        # L 通道 (亮度) 混合，a/b 通道 (色彩) 直接继承
        dest_lab[..., 0] = src_lab[..., 0] * 0.2 + dest_lab[..., 0] * 0.8
        dest_lab[..., 1] = src_lab[..., 1]
        dest_lab[..., 2] = src_lab[..., 2]
        
        dest_lab = np.clip(dest_lab, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(dest_lab, cv2.COLOR_LAB2BGR)
        return out
    except Exception as e:
        print(f"色彩迁移失败: {e}")
        return dest

# ------------------ 4. 自适应推理（裁剪边缘 + 恢复原尺寸） ------------------
def infer_adaptive(model, input_img):
    """
    对图像进行缩放、推理、裁剪边缘，并将结果拉伸回原始分辨率。
    """
    global MAX_SIDE_SIZE, TOTAL_CROP_MARGIN
    
    # 1. 记录原始尺寸 (h_orig x w_orig)
    h_orig, w_orig = input_img.shape[:2]
    
    # 2. 推理：itot 中已完成缩放和 PAD 填充
    tensor = itot(input_img, max_size=MAX_SIDE_SIZE) 
    gen_tensor = model(tensor)
    
    # 3. 转换回图像
    generated_image_padded = ttoi(gen_tensor)

    # 4. 使用 TOTAL_CROP_MARGIN 进行裁剪 (移除不稳定边缘)
    h_current, w_current = generated_image_padded.shape[:2]
    
    # 修正：使用更大的裁剪边距
    generated_image = generated_image_padded[
        TOTAL_CROP_MARGIN : h_current - TOTAL_CROP_MARGIN, 
        TOTAL_CROP_MARGIN : w_current - TOTAL_CROP_MARGIN
    ]
    
    # 5. 【核心修改：拉伸回原始分辨率】
    if generated_image.shape[:2] != (h_orig, w_orig):
        # 使用 INTER_CUBIC 进行高质量上采样
        # 注意 cv2.resize 接收 (width, height)
        generated_image = cv2.resize(generated_image, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    
    # 6. 返回结果
    return generated_image