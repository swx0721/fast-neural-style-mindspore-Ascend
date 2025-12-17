# stylize.py - 最终修正版 (复现训练采样图的纯净效果 + 移除不必要后处理)
import mindspore as ms
from mindspore import Tensor, context, ops
# 导入正确的模块名称
import transformer
import utils
import os
import time
from transformer import TransformerNet # 从 transformer.py 导入
import cv2
import numpy as np

# ------------------ GLOBAL SETTINGS ------------------
# 请将此路径替换为您实际训练得到的 checkpoint 路径
STYLE_TRANSFORM_PATH = "models1/sumiao_checkpoint_4000.ckpt" 
PRESERVE_COLOR = True # <<< 关键修正 1：强制关闭色彩迁移
target_device = "Ascend"
OUTPUT_DIR = "images/results1/"#原images/results
context.set_context(mode=context.GRAPH_MODE, device_target=target_device)

# ------------------ 单图风格迁移 ------------------
def stylize():
    global STYLE_TRANSFORM_PATH
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 加载网络（保持不变）
    while True:
        try:
            # 默认 TransformerNet() 是 standard mode，如果训练时使用了 high_res_mode 需要传入对应参数
            net = TransformerNet() 
            # 检查模型文件是否存在
            if not os.path.exists(STYLE_TRANSFORM_PATH):
                 print(f"❌ 模型文件未找到: {STYLE_TRANSFORM_PATH}")
                 STYLE_TRANSFORM_PATH = input("请输入正确的 checkpoint 路径：").strip()
                 continue
                 
            param_dict = ms.load_checkpoint(STYLE_TRANSFORM_PATH)
            ms.load_param_into_net(net, param_dict)
            net.set_train(False)
            print("✅ Transformer Network Loaded Successfully.\n")
            break
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            STYLE_TRANSFORM_PATH = input("请输入正确的 checkpoint 路径：").strip()
            continue

    # 2. 推理循环
    while True:
        try:
            print("\n🎨 Stylize Image~ 输入 Ctrl+C 退出程序")
            content_image_path = input("请输入内容图像路径： ").strip()
            if content_image_path == "" or not os.path.isfile(content_image_path):
                print("⚠ 无效路径，请重新输入。")
                continue

            content_image = utils.load_image(content_image_path)
            if content_image is None:
                print("❌ 图像加载失败，请检查格式（支持jpg/png）。")
                continue

            starttime = time.time()
            h, w = content_image.shape[:2]
            
            print(f"📸 检测到图像分辨率 ({w}x{h})，启用无伪影自适应推理...")
            # 核心推理：使用 utils 中的 infer_adaptive，返回 BGR numpy [0, 255]
            generated_image = utils.infer_adaptive(net, content_image)

            # -------------------- 后处理 --------------------
            if PRESERVE_COLOR:
                generated_image = utils.transfer_color(content_image, generated_image)
            # ❗ 关键修正 2：移除所有不必要的色彩校准代码
            # ----------------------------------------------------

            output_filename = "styled_" + os.path.basename(content_image_path)
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            utils.saveimg(generated_image, output_path)

            print(f"✅ 风格迁移完成，结果保存至: {output_path}")
            print(f"⏱ 推理耗时: {time.time() - starttime:.2f} 秒\n")
            
        except KeyboardInterrupt:
            print("\n程序退出。")
            break
        except Exception as e:
            print(f"发生错误: {e}")

# ------------------ 文件夹批量风格迁移 ------------------
def stylize_folder(content_folder, save_folder=None, batch_size=1):
    if save_folder is None:
        save_folder = os.path.join(content_folder, "styled_results_ascend")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    net = TransformerNet()
    param_dict = ms.load_checkpoint(STYLE_TRANSFORM_PATH)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    image_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [
        os.path.join(content_folder, f)
        for f in os.listdir(content_folder)
        if f.lower().endswith(image_ext)
    ]

    if not image_paths:
        print("⚠ 文件夹内未检测到图像文件")
        return

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        for img_path in batch_paths:
            content_image = utils.load_image(img_path)
            if content_image is None:
                print(f"❌ 跳过无效图像: {img_path}")
                continue
            
            h, w = content_image.shape[:2]
            print(f"📸 批量处理: {os.path.basename(img_path)} ({w}x{h})")
            generated_image = utils.infer_adaptive(net, content_image)
            
            if PRESERVE_COLOR: 
                generated_image = utils.transfer_color(content_image, generated_image)
            
            # ❗ 关键修正 3：移除批量处理中的不必要的色彩校准代码
            
            output_filename = "styled_" + os.path.basename(img_path)
            output_path = os.path.join(save_folder, output_filename)
            utils.saveimg(generated_image, output_path)
            print(f"✅ 保存至: {output_path}")

if __name__ == '__main__':
    stylize()