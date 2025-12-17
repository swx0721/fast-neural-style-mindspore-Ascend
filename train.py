# train.py - MindSpore Ascend 最终兼容版 (已修复 TotalVariation 兼容性错误及 optimizer 属性错误)
import mindspore as ms
from mindspore import nn, ops, Tensor, context, dtype as mstype
from mindspore.dataset import GeneratorDataset
# 引入 T (transforms) 用于 Compose
import mindspore.dataset.transforms as T 
# 引入 V (vision) 用于 Resize/CenterCrop
import mindspore.dataset.vision as V
import numpy as np
import random
import time
import os
import cv2

# ❗ 关键修改 1：引入学习率调度器
from mindspore.nn import cosine_decay_lr 

import transformer
import vgg
import utils

# ------------------ GLOBAL SETTINGS ------------------
TRAIN_IMAGE_SIZE = 256
DATASET_PATH = "train2017"  # 训练数据集
NUM_EPOCHS = 1
# ❗ 参数更新：新的风格图像路径
STYLE_IMAGE_PATH = "images/oil_painting.jpg"
BATCH_SIZE = 4
# ❗ 优化：降低内容权重、降低总风格权重、提高初始学习率以稳定 Loss
CONTENT_WEIGHT = 16.0  
STYLE_WEIGHT = 30.0   
ADAM_LR = 2e-4        # (原 1e-3) 初始学习率，配合动态衰减

# ❗ NEW: TV Loss 权重 (用于平滑图像，消除彩色条纹伪影)
TV_WEIGHT = 1e-2 # 这是一个经验值，用于轻微正则化平滑度

SAVE_MODEL_PATH = "models/"
SAVE_IMAGE_PATH = "images/results/"
SAVE_MODEL_EVERY = 250
PRINT_GRAD_EVERY = 50
SEED = 35
PLOT_LOSS = 1

# ❗ 参数更新：新的固定采样内容图路径
FIXED_SAMPLE_CONTENT_PATH = "images/face.jpg" 

# MindSpore Ascend 适配参数
GRAD_CLIP_VALUE = 1.0 
NUM_WORKERS = 64 # 增加数据加载并行数

# ------------------ Device Setting ------------------
target_device = "Ascend"
context.set_context(mode=context.GRAPH_MODE, device_target=target_device)
# ------------------ Seed ------------------
ms.common.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ------------------ Dataset ------------------
class CustomDataset:
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        img = utils.load_image(path)
        if img is None:
            # 随机返回一个有效的样本，避免训练中断
            return self.__getitem__(random.randint(0, len(self.image_paths) - 1))
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # 这里的 transform 接收 np.ndarray
            img_rgb = self.transform(img_rgb)
            
        # 将 numpy 转换成 MindSpore Tensor，并转到 [-1, 1] 范围
        img_tensor = Tensor(img_rgb, mstype.float32)
        # HWC -> CHW (MindSpore/PyTorch 格式)
        img_tensor = ops.transpose(img_tensor, (2, 0, 1))
        # 归一化到 [-1, 1]
        img_tensor = img_tensor / 127.5 - 1.0
        
        return img_tensor

def create_dataloader(dataset_path, image_size, batch_size, num_workers):
    # 修正：使用 T.Compose 替换 V.Compose
    transform = T.Compose([
        V.Resize(image_size),
        V.CenterCrop(image_size)
    ])
    
    dataset = CustomDataset(dataset_path, transform=transform)
    
    # 确保 drop_remainder=True 以在 GRAPH_MODE 下保持固定的 Batch Size
    data_loader = GeneratorDataset(
        dataset, 
        column_names=["content_image"],
        shuffle=True, 
        num_parallel_workers=num_workers,
        max_rowsize=32  # 增加 max_rowsize 避免内存警告
    )
    data_loader = data_loader.batch(batch_size, drop_remainder=True)
    # ❗ 关键修改：返回 dataset 对象
    return data_loader, len(dataset), dataset 

# ------------------ Loss Network (核心修改) ------------------
class StyleTransferLoss(nn.Cell):
    def __init__(self, transformer_net, content_weight, style_weight, tv_weight):
        super().__init__()
        self.transformer = transformer_net
        self.vgg = vgg.VGG19_Feature()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight 

        # ❗ 优化：新的分层风格权重 (Style Layer Weights) - 保持用户当前配置
        self.style_layer_weights = {
            'relu1_2': 0.5, 
            'relu2_2': 1.0,
            'relu3_4': 1.0,
            'relu4_4': 0.8,
            'relu5_4': 0.2  
        }
        
        self.square = ops.Square()
        self.reduce_mean = ops.ReduceMean()
        
        # 修复：用于手动计算 TV Loss
        self.abs = ops.Abs()
        self.reduce_sum = ops.ReduceSum()

        # 预先计算风格图像的特征和 Gram 矩阵
        style_image = utils.load_image(STYLE_IMAGE_PATH)
        
        # ❗ 核心修改：CLAHE 预处理风格图，增强高亮/高暗鲁棒性
        if style_image is None:
             raise FileNotFoundError(f"❌ 严重错误: 无法加载风格图像，请检查路径: {STYLE_IMAGE_PATH}")
        
        # 1. 缩放风格图，避免 MindSpore 内存溢出 (保持原有逻辑)
        TARGET_STYLE_SIZE = 720 
        h, w = style_image.shape[:2]
        if max(h, w) > TARGET_STYLE_SIZE:
            ratio = TARGET_STYLE_SIZE / max(h, w)
            new_h = int(h * ratio)
            new_w = int(w * ratio)
            style_image = cv2.resize(style_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"✅ 风格图已缩放至: {new_w}x{new_h}，以避免 MindSpore 内存溢出。")
            
        # 2. BGR -> RGB，准备进行 CLAHE
        style_image_rgb = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
        
        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        style_img_yuv = cv2.cvtColor(style_image_rgb, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(style_img_yuv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y_clahe = clahe.apply(y)
        yuv_clahe = cv2.merge([y_clahe, u, v])
        style_img_clahe = cv2.cvtColor(yuv_clahe, cv2.COLOR_YUV2RGB)
        
        # 4. 转换到 Tensor
        style_img_clahe = style_img_clahe.astype(np.float32)
        style_img_clahe = style_img_clahe / 255.0 * 2.0 - 1.0 
        style_tensor = Tensor(np.expand_dims(style_img_clahe.transpose(2, 0, 1), 0), mstype.float32)
        
        self.style_features = self.vgg(style_tensor) 
        
        self.style_grams = {}
        for name, feature in self.style_features.items():
            gram_matrix = utils.gram(feature)
            self.style_grams[name] = ops.squeeze(gram_matrix, axis=0) 
            
    def _mse_loss(self, pred, target):
        """手动计算 MSE loss：Mean(Square(Pred - Target))"""
        return self.reduce_mean(self.square(pred - target))

    def construct(self, content_image):
        generated_image = self.transformer(content_image)
        
        # 1. Content Loss
        content_features = self.vgg(content_image)
        generated_features = self.vgg(generated_image)
        
        # Content Loss Layer: relu3_4
        content_loss = self._mse_loss(content_features['relu2_2'], generated_features['relu2_2'])
        
        # 2. Style Loss
        style_loss = Tensor(0.0, mstype.float32)
        for layer in ['relu1_2','relu2_2','relu3_4','relu4_4','relu5_4']:
            gen_gram = utils.gram(generated_features[layer])
            style_gram = self.style_grams[layer]
            C = style_gram.shape[0] 
            broadcast_shape = (gen_gram.shape[0], C, C) 
            style_gram_batched = ops.broadcast_to(style_gram, broadcast_shape)
            
            # 累加 Style Loss 时应用分层权重
            layer_loss = self._mse_loss(gen_gram, style_gram_batched)
            style_loss += layer_loss * self.style_layer_weights[layer] 

        # ❗ 手动计算 TV Loss (L1 norm Total Variation)
        # 计算高方向的差值 (H, W-1)
        tv_loss_h = self.abs(generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :])
        # 计算宽方向的差值 (H-1, W)
        tv_loss_w = self.abs(generated_image[:, :, :, 1:] - generated_image[:, :, :, :-1])
        # 将所有差值的绝对值求和，作为 TV Loss
        tv_loss = self.reduce_sum(tv_loss_h) + self.reduce_sum(tv_loss_w)

        # 3. Total Loss (加入 TV Loss)
        total_loss = (self.content_weight * content_loss + 
                      self.style_weight * style_loss +
                      self.tv_weight * tv_loss)
        
        # 4. 返回 tv_loss
        return total_loss, content_loss, style_loss, tv_loss, generated_image

# ------------------ Training ------------------

# ❗ 关键新增：固定采样图预处理函数
def _load_and_preprocess_sample_image(path, size):
    """加载并预处理固定采样图，使其与 dataloader 输出格式一致：[1, C, H, W], [-1, 1]"""
    # 1. 加载图像
    raw_sample_img = utils.load_image(path)
    if raw_sample_img is None:
        raise FileNotFoundError(f"无法加载采样内容图，请检查路径: {path}")

    # 2. BGR -> RGB
    img_rgb = cv2.cvtColor(raw_sample_img, cv2.COLOR_BGR2RGB)
    
    # 3. 仿照 dataloader 的 Resize/CenterCrop
    h, w, _ = img_rgb.shape
    ratio = size / min(h, w)
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    start_h = (new_h - size) // 2
    start_w = (new_w - size) // 2
    img_cropped = img_resized[start_h:start_h + size, start_w:start_w + size, :]
    
    # 4. HWC -> CHW, Normalize to [-1, 1], Expand_dims
    fixed_sample_tensor = Tensor(img_cropped, mstype.float32)
    fixed_sample_tensor = ops.transpose(fixed_sample_tensor, (2, 0, 1)) # CHW
    fixed_sample_tensor = fixed_sample_tensor / 127.5 - 1.0
    fixed_sample_tensor = ops.expand_dims(fixed_sample_tensor, 0) # [1, C, H, W]
    
    return fixed_sample_tensor


def train():
    # 1. Prepare Data
    data_loader, total_data_size, raw_dataset = create_dataloader(DATASET_PATH, TRAIN_IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)
    steps_per_epoch = data_loader.get_dataset_size()
    print(f"Dataset Size: {total_data_size}")

    # 计算总训练步数
    TOTAL_TRAIN_STEPS = steps_per_epoch * NUM_EPOCHS
    print(f"Total Training Steps: {TOTAL_TRAIN_STEPS}")

    # 2. Prepare Fixed Sample Image
    sample_img_path = FIXED_SAMPLE_CONTENT_PATH
    
    if not os.path.exists(sample_img_path) and raw_dataset.image_paths:
         sample_img_path = raw_dataset.image_paths[0] 
         print(f"⚠️ 警告: 未找到固定的采样内容图。使用数据集中的第一张图作为采样图: {sample_img_path}")
    
    try:
        fixed_sample_tensor = _load_and_preprocess_sample_image(sample_img_path, TRAIN_IMAGE_SIZE)
        print(f"✅ 已加载固定采样内容图: {os.path.basename(sample_img_path)}")
    except FileNotFoundError as e:
         print(f"❌ 严重错误: {e}")
         return
    
    # 3. Network and Loss
    global TransformerNetwork
    TransformerNetwork = transformer.TransformerNet(high_res_mode=False) 
    
    LossNetwork = StyleTransferLoss(TransformerNetwork, CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT)
    LossNetwork.set_train() 
    
    # 4. Optimizer and Training Cell 
    lr = cosine_decay_lr(
        min_lr=0.0,
        max_lr=ADAM_LR,
        total_step=TOTAL_TRAIN_STEPS,
        step_per_epoch=steps_per_epoch,
        decay_epoch=NUM_EPOCHS
    )
    
    optimizer = nn.Adam(TransformerNetwork.trainable_params(), learning_rate=lr) 
    
    class TrainOneStepCell(nn.Cell):
        def __init__(self, net, optimizer, grad_clip):
            super().__init__()
            self.net = net
            self.optimizer = optimizer 
            self.grad_fn = ops.value_and_grad(self.net, None, self.optimizer.parameters, has_aux=True) 
            self.clip_by_norm = nn.ClipByNorm(axis=None)
            self.hyper_map = ops.HyperMap()
            grad_clip_tensor = ops.scalar_to_tensor(grad_clip)
            num_params = len(self.optimizer.parameters)
            self.clip_norm_tensors = tuple([grad_clip_tensor] * num_params)

        def construct(self, content_image):
            (total_loss, content_loss, style_loss, tv_loss, generated_image), grads = self.grad_fn(content_image)
            grads = self.hyper_map(self.clip_by_norm, grads, self.clip_norm_tensors)
            total_loss = ops.depend(total_loss, self.optimizer(grads))
            return total_loss, content_loss, style_loss, tv_loss, generated_image

    train_net = TrainOneStepCell(LossNetwork, optimizer, GRAD_CLIP_VALUE)
    
    # 5. Start Training
    start_time = time.time()
    batch_count = 0
    
    # --- 初始化所有 Loss 历史记录列表 ---
    total_loss_history = []
    content_loss_history = []
    style_loss_history = []
    tv_loss_history = [] # NEW
    
    for epoch in range(NUM_EPOCHS):
        for step, (content_batch,) in enumerate(data_loader.create_tuple_iterator()):
            current_total, current_content, current_style, current_tv, generated_batch = train_net(content_batch)
            
            # 转换为 numpy 格式用于记录和绘图
            current_total = current_total.asnumpy()
            current_content = current_content.asnumpy()
            current_style = current_style.asnumpy()
            current_tv = current_tv.asnumpy()
            
            batch_count += 1
            batch_in_epoch = step + 1

            # 记录数据
            total_loss_history.append(current_total)
            content_loss_history.append(current_content)
            style_loss_history.append(current_style)
            tv_loss_history.append(current_tv) # NEW

            # 打印日志
            if batch_count % PRINT_GRAD_EVERY == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{batch_in_epoch}/{steps_per_epoch}] "
                      f"Loss: {current_total:.2f} (Content: {current_content:.2f}, Style: {current_style:.2f}, TV: {current_tv:.6f})")
            
            # 每 250 步（SAVE_MODEL_EVERY）执行：保存模型、采样、绘图
            if batch_count % SAVE_MODEL_EVERY == 0 or (epoch == NUM_EPOCHS - 1 and step == steps_per_epoch - 1):
                os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
                os.makedirs(SAVE_IMAGE_PATH, exist_ok=True)
                
                # A. 保存模型
                checkpoint_path = os.path.join(SAVE_MODEL_PATH, f"checkpoint_{batch_count}.ckpt")
                ms.save_checkpoint(TransformerNetwork, checkpoint_path)
                
                # B. 关键修改：实时绘制并保存所有 Loss 曲线
                if PLOT_LOSS:
                    plot_save_path = os.path.join(SAVE_IMAGE_PATH, "loss_curve_latest.png")
                    utils.plot_losses(
                        total_loss_history, 
                        content_loss_history, 
                        style_loss_history, 
                        tv_loss_history, # 传入四个参数
                        save_path=plot_save_path
                    )
                
                # C. 使用固定采样图进行推理
                TransformerNetwork.set_train(False) 
                sample_tensor = TransformerNetwork(fixed_sample_tensor) 
                TransformerNetwork.set_train(True) 
                
                # 保存采样图像
                sample_image = utils.ttoi(sample_tensor)
                utils.saveimg(sample_image, os.path.join(SAVE_IMAGE_PATH, f"sample_fixed_{batch_count}.png"))
                print(f"📊 Step {batch_count}: 模型已保存，Loss 曲线已更新。")

    stop_time = time.time()
    print(f"Done Training! Time elapsed: {stop_time - start_time:.2f} seconds")
    
    # Final Save
    TransformerNetwork.set_train(False)
    final_path = os.path.join(SAVE_MODEL_PATH, f"final_{os.path.basename(STYLE_IMAGE_PATH).split('.')[0]}.ckpt")
    ms.save_checkpoint(TransformerNetwork, final_path)
    print(f"Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    train()