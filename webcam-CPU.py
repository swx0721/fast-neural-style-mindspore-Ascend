# webcam.py (MindSpore CPU 最终修正版 - 优化实时延迟)
import cv2
import transformer
import utils
import mindspore as ms
from mindspore import context
import time
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["KMP_BLOCKTIME"] = "0"

# ------------------ GLOBAL SETTINGS ------------------
STYLE_TRANSFORM_PATH = "models/mosaic_checkpoint_24000.ckpt"
PRESERVE_COLOR = False
WIDTH = 1080
HEIGHT = 720

context.set_context(
    mode=context.GRAPH_MODE,
    device_target="CPU",
)

def webcam(style_transform_path, width=WIDTH, height=HEIGHT):
    # 1. 加载 Transformer Network
    net = transformer.TransformerNet(high_res_mode=False)
    param_dict = ms.load_checkpoint(style_transform_path)
    ms.load_param_into_net(net, param_dict)
    net.set_train(False)

    # CPU上尝试半精度加速
    net.to_float(ms.float16)
    dtype = ms.float16

    # 2. 设置摄像头
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    start_time = time.time()
    frame_id=0
    fps=0

    # 3. 主循环
    while True:
        ret_val, img = cam.read()

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (width//3, height//3),interpolation=cv2.INTER_AREA)

        # utils.itot 负责缩放和归一化到 TRAIN_IMAGE_SIZE
        content_tensor = utils.itot(img)
        generated_tensor = net(content_tensor)
        # utils.ttoi 负责反归一化和 Tensor->BGR numpy
        generated_image = utils.ttoi(generated_tensor)
        
        # 保持纯净输出
        if PRESERVE_COLOR:
            generated_image = utils.transfer_color(img, generated_image)
        
        # 计算 FPS
        frame_id += 1
        if frame_id % 10 == 0:
            end_time = time.time()
            fps = 10.0 / (end_time - start_time)
            start_time = end_time

        # 显示
        generated_image = cv2.resize(generated_image, (width, height),interpolation=cv2.INTER_LINEAR)
        cv2.putText(
            generated_image, 
            f"FPS: {fps:.2f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        cv2.imshow('Real-time Style Transfer Demo', generated_image)
        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam(STYLE_TRANSFORM_PATH)