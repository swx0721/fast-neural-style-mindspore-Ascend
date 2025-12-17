# video_ascend.py (视频离线处理 - Ascend NPU 优化版)
import utils
import transformer
import cv2
import os
import time
import mindspore as ms
from mindspore import context, ops
# 假设 stylize.py 中已经有用于批量处理的函数 stylize_folder
from stylize import stylize_folder 
# 导入 TransformerNetwork，确保模型能被正确加载
from transformer import TransformerNetwork 

# ------------------ GLOBAL SETTINGS ------------------
VIDEO_NAME = "input_video.mp4"
FRAME_SAVE_PATH = "frames/"
STYLE_FRAME_SAVE_PATH = "style_frames/"
STYLE_VIDEO_NAME = "styled_output.mp4"
STYLE_PATH = "transforms/mosaic.ckpt" 
BATCH_SIZE = 16 # Ascend 上可以尝试更高的批量大小以提升吞吐量

# 🎯 MindSpore Ascend 适配：设置 GRAPH_MODE 
target_device = "Ascend"
context.set_context(mode=context.GRAPH_MODE, device_target=target_device) 

# 辅助常量
FRAME_BASE_FILE_NAME = "frame"
FRAME_BASE_FILE_TYPE = ".jpg"

# ------------------ 辅助函数 (保持不变) ------------------
def getInfo(video_path):
    """提取视频信息"""
    vidcap = cv2.VideoCapture(video_path)
    # ... (保持不变)
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    fps =  vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    return height, width, fps

def getFrames(video_path):
    """提取视频所有帧并保存"""
    # ... (保持不变)
    
def makeVideo(frames_path, save_name, width, height, fps):    
    """将风格化后的帧合并成视频"""
    # ... (保持不变)

# ------------------ 主函数 ------------------
def video_transfer(video_path, style_path):
    print("OpenCV {}".format(cv2.__version__))
    starttime = time.time()
    
    # 提取视频信息
    H, W, fps = getInfo(video_path)
    print("Height: {} Width: {} FPS: {}".format(H, W, fps))

    # 提取所有帧
    print("Extracting video frames...")
    getFrames(video_path)
    
    # 🎯 对帧目录进行批量风格化 (利用 Ascend NPU 加速)
    print("Starting batch style transfer on Ascend NPU...")
    # 假设 stylize_folder 接受 (content_folder, save_folder, style_path, batch_size)
    # 我们将 FRAME_SAVE_PATH 作为输入 content_folder
    # stylize_folder 内部会加载模型并执行推理
    stylize_folder(FRAME_SAVE_PATH, STYLE_FRAME_SAVE_PATH, style_path, BATCH_SIZE)
    
    # 重新合并成视频
    print("Re-assembling video frames...")
    makeVideo(STYLE_FRAME_SAVE_PATH, STYLE_VIDEO_NAME, W, H, fps)

    endtime = time.time()
    print(f"✅ Video style transfer completed. Total time: {endtime - starttime:.2f} seconds")

if __name__ == '__main__':
    video_transfer(VIDEO_NAME, STYLE_PATH)