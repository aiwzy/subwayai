import mss
import cv2
import numpy as np
import os
import time
import keyboard
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pynput.keyboard import Controller
from threading import Thread
from queue import Queue
from datetime import datetime


# ====================== 模型定义 ======================
#注意力机制
class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        mid_channels = int(in_channels * se_ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.Hardswish(),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.squeeze(x).view(b, c, 1, 1)
        se = self.excitation(se)
        return x * se.expand_as(x)


class TimeFocusedModel(nn.Module):
    def __init__(self, num_classes=5, time_importance=[0.5, 0.3, 3.0], dropout_p=0.3):
        super().__init__()
        self.time_importance = nn.Parameter(
            torch.tensor(time_importance, dtype=torch.float32),
            requires_grad=True
        )

        # 空间卷积
        self.spatial_encoder = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Hardswish(),

            InvertedResidual(32, 32, stride=2, expand_ratio=6),

            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 64, stride=2, expand_ratio=6),

            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 128, stride=2, expand_ratio=6),

            InvertedResidual(128, 128, stride=1, expand_ratio=6),
            InvertedResidual(128, 256, stride=2, expand_ratio=6),

            InvertedResidual(256, 256, stride=2, expand_ratio=4),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Conv2d(256, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1)
        )

        # 时间卷积
        self.time_raw_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),

            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),

            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),

            InvertedResidual(32, 64, stride=2, expand_ratio=6),

            nn.AdaptiveAvgPool2d(1)
        )

        # 时间序列1D卷积
        self.time_conv = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.Hardswish(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 128, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(),
            nn.Dropout(0.2),

            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.1),

            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # 空间特征提取
        spatial_feat = self.spatial_encoder(x[:, 2:3, :, :])
        spatial_feat = spatial_feat.flatten(1)
        spatial_feat = spatial_feat * self.time_importance[2]

        # 时间特征提取
        time_raw_feats = []
        for t in range(3):
            raw_feat = self.time_raw_encoder(x[:, t:t + 1, :, :])
            raw_feat = raw_feat.flatten(1)
            raw_feat = raw_feat * self.time_importance[t]
            time_raw_feats.append(raw_feat)

        time_sequence = torch.stack(time_raw_feats, dim=2)
        time_feat = self.time_conv(time_sequence).squeeze(2)

        # 特征融合与分类
        combined_feat = torch.cat([spatial_feat, time_feat], dim=1)
        return self.classifier(combined_feat)


# 逆残差块
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se_ratio=0.25):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        expand_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, expand_channels, kernel_size=1),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(),

            nn.Conv2d(expand_channels, expand_channels, kernel_size=3, stride=stride, padding=1, groups=expand_channels),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(),

            SEBlock(expand_channels, se_ratio=se_ratio),

            nn.Conv2d(expand_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ====================== 游戏控制类 ======================
class GameController:
    def __init__(self):
        self.keyboard = Controller()
        self.action_map = {0: None, 1: 's', 2: 'a', 3: 'd', 4: 'w'}

    def perform_action(self, action):
        key = self.action_map.get(action, None)
        if key:
            self.keyboard.press(key)
            time.sleep(0.4)  # 按键持续时间
            self.keyboard.release(key)
            print(f"执行键盘动作: {key}")


# ====================== 截图与推理整合 ======================
start_time = datetime.now()
class RealTimeController:
    def __init__(self, output_path="sc/"):
        self.output_path = output_path
        self.target_size = (128, 128)
        self.auto_interval = 0.05
        self.max_cache = 5
        self.auto_screenshots = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(r"C:\Users\bxd\Desktop\CurrentBestModel.pth")
        self.controller = GameController()
        self.tensor_queue = Queue()  # 用于传递预处理后的张量
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self._init_dirs()

    def _init_dirs(self):
        # 由于不保存NPY文件，只需要确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)

    def load_model(self, model_path):
        model = TimeFocusedModel(num_classes=5)
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def capture_loop(self):
        """截图线程主函数"""
        sc = mss.mss()
        region = {"top": 0, "left": 128, "width": 896, "height": 768}
        last_save = time.time()

        while not keyboard.is_pressed('k'):
            # 捕获屏幕并转换为灰度图
            img = np.array(sc.grab(region))
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            # 维护截图缓存
            self.auto_screenshots.append(gray)
            if len(self.auto_screenshots) > self.max_cache:
                # 生成合成图
                composite = np.stack([
                    self.auto_screenshots[0],  # t-3帧
                    self.auto_screenshots[2],  # t-1帧
                    self.auto_screenshots[-1]  # 当前帧
                ], axis=0)

                # 预处理并放入队列
                processed_tensor = self.preprocess_tensor(composite)
                self.tensor_queue.put(processed_tensor)

                # 移除最早的帧
                self.auto_screenshots.pop(0)

            # 控制截图频率
            time.sleep(max(0, self.auto_interval - (time.time() - last_save)))
            last_save = time.time()

    def preprocess_tensor(self, tensor):
        """预处理张量: 调整尺寸并转换为PyTorch张量（使用双三次插值）"""
        # 使用cv2.INTER_CUBIC参数指定双三次插值算法
        resized = np.array([cv2.resize(frame, self.target_size, interpolation=cv2.INTER_CUBIC) for frame in tensor])

        # 转换为PyTorch张量并归一化
        tensor = torch.from_numpy(resized).float().unsqueeze(0).to(self.device)  # (1, 3, 128, 128)
        return self.transform(tensor)
    def inference_loop(self):
        """推理线程主函数"""
        while not keyboard.is_pressed('k'):
            if not self.tensor_queue.empty():
                inputs = self.tensor_queue.get()
                try:
                    # 直接推理，不通过文件
                    with torch.no_grad():
                        output = self.model(inputs)
                        action = torch.argmax(output, dim=1).item()
                    self.controller.perform_action(action)
                    print(f"执行动作: {action}")
                except Exception as e:
                    print(f"推理失败: {e}")

    def run(self):
        """启动双线程"""
        print("系统启动，按ESC退出")
        capture_thread = Thread(target=self.capture_loop, daemon=True)
        inference_thread = Thread(target=self.inference_loop, daemon=True)

        capture_thread.start()
        inference_thread.start()

        while not keyboard.is_pressed('esc'):
            time.sleep(0.1)
        print("系统退出")


# ====================== 主程序入口 ======================
if __name__ == "__main__":
    rt_controller = RealTimeController(output_path=r"C:\Users\bxd\Desktop\sc")
    rt_controller.run()
