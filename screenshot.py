import mss
import cv2
import numpy as np
import time
import keyboard
import os
from datetime import datetime  # 新增导入


class ScreenCapture:
    def __init__(self, region=None, output_path="screenshots/"):
        """初始化屏幕截图工具"""
        self.region = region or self.get_fullscreen_region()
        self.output_path = output_path
        self.frame = None
        self.last_time = time.time()
        self.auto_interval = 0.05  # 自动截图间隔（0.05秒）
        # 新增：自动截图缓存（最多保存4张）
        self.auto_screenshots = []
        self.max_cache_size = 4

        # 定义5个保存目录
        self.directories = {
            'auto': os.path.join(output_path, 'auto'),
            'up': os.path.join(output_path, 'up'),
            'down': os.path.join(output_path, 'down'),
            'left': os.path.join(output_path, 'left'),
            'right': os.path.join(output_path, 'right')
        }

        # 创建所有目录
        for dir_path in self.directories.values():
            os.makedirs(dir_path, exist_ok=True)

        # 按键映射
        self.key_mapping = {
            'w': 'up',
            's': 'down',
            'a': 'left',
            'd': 'right'
        }
        self.key_states = {key: False for key in self.key_mapping}

        # 自动截图计时
        self.last_auto_save = time.time()

    def get_fullscreen_region(self):
        """自动获取全屏区域"""
        with mss.mss() as sct:
            return sct.monitors[0]  # 全屏幕区域

    def capture_frame(self):
        """捕获当前屏幕帧并转换为灰度图"""
        with mss.mss() as sct:
            img = sct.grab(self.region)
            # 直接转换为灰度图
            self.frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2GRAY)
            return self.frame

    def save_screenshot(self, direction='auto', quality=85):
        """保存截图到指定目录"""
        if direction not in self.directories:
            print(f"无效方向: {direction}")
            return None

        # 使用datetime.now()获取包含微秒的时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dir_path = self.directories[direction]
        filename = os.path.join(dir_path, f"{direction}_{timestamp}.jpg")

        cv2.imwrite(filename, self.frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"截图保存到 {direction}: {filename}")
        return filename

    def cache_auto_screenshot(self):
        """仅缓存自动截图，不保存到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 添加到缓存
        self.auto_screenshots.append({
            'frame': self.frame.copy(),  # 保存副本
            'timestamp': timestamp
        })

        # 保持缓存大小不超过上限
        if len(self.auto_screenshots) > self.max_cache_size:
            self.auto_screenshots.pop(0)

        return timestamp

    def create_composite_image(self, base_frame, direction):
        """创建张量级合成（三张图作为三个通道，维度为 [3, H, W]）"""
        cache_size = len(self.auto_screenshots)

        if cache_size < 4:
            print(f"缓存不足，无法创建合成图")
            return None, None, None

        # 获取前第二张和前第四张自动截图
        second_prev = self.auto_screenshots[-2]
        fourth_prev = self.auto_screenshots[-4]

        # 调整尺寸为相同大小
        h, w = base_frame.shape[:2]
        img1 = cv2.resize(fourth_prev['frame'], (w, h))  # 尺寸变为 (h, w)（H, W）
        img2 = cv2.resize(second_prev['frame'], (w, h))
        img3 = base_frame  # 尺寸为 (h, w)

        # 张量级合成（维度改为 [3, H, W]）
        composite_tensor = np.stack([img1, img2, img3], axis=0)  # [3, H, W]

        # 保存时需要转换为 OpenCV 支持的 [H, W, 3] 格式（通道在后）
        composite_for_save = np.transpose(composite_tensor, (1, 2, 0))  # 转为 [H, W, 3]
        composite_for_save = cv2.cvtColor(composite_for_save, cv2.COLOR_GRAY2BGR)  # 灰度图转三通道

        # 保存为 PNG（支持多通道，用于人类查看）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dir_path = self.directories[direction]
        img_filename = os.path.join(dir_path, f"tensor_{direction}_{timestamp}.png")
        cv2.imwrite(img_filename, composite_for_save)
        print(f"图像保存到: {img_filename} (维度: {composite_for_save.shape})")

        # 额外保存原始张量（[3, H, W] 格式，用于模型处理）
        tensor_filename = os.path.join(dir_path, f"tensor_{direction}_{timestamp}.npy")
        np.save(tensor_filename, composite_tensor)
        print(f"原始张量保存到: {tensor_filename} (维度: {composite_tensor.shape})")

        # 返回原始张量、预览图和文件名
        preview = np.hstack([img1, img2, img3])  # 水平拼接仍为 [H, 3W]
        return composite_tensor, preview, (img_filename, tensor_filename)

    def load_composite_tensor(file_path):
        """加载保存的原始张量（恢复 [3, H, W] 维度）"""
        try:
            tensor = np.load(file_path)
            print(f"成功加载张量，维度: {tensor.shape}")
            return tensor
        except Exception as e:
            print(f"加载张量失败: {e}")
            return None

    def show_frame(self, window_name="Screen Capture"):
        """实时显示当前帧（灰度图）"""
        if self.frame is not None:
            fps = 1.0 / (time.time() - self.last_time)
            cv2.putText(self.frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 255, 2), 2)

            # 显示操作提示
            cv2.putText(self.frame, "自动保存: 每秒1张 (auto目录)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ( 0.7, 255, 2), 2)
            cv2.putText(self.frame, "按键保存: W=上 S=下 A=左 D=右", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0.7, 255, 2), 2)

            cv2.imshow(window_name, self.frame)
            self.last_time = time.time()

    def run(self, show=True):
        """启动双模式截图（自动+按键）"""
        print("截图程序启动 - 双模式运行:")
        print("  自动模式: 每0.05秒保存1张到 auto 目录")
        print("  按键模式: W/S/A/D 分别保存到 up/down/left/right 目录")
        print("按 ESC 退出程序")

        while not keyboard.is_pressed('esc'):
            self.capture_frame()

            # 处理自动保存（每秒0.05次）
            if time.time() - self.last_auto_save >= self.auto_interval:
                self.cache_auto_screenshot()  # 仅缓存，不保存文件
                self.last_auto_save = time.time()

            # 处理按键保存合成图
            for key, direction in self.key_mapping.items():
                if keyboard.is_pressed(key) and not self.key_states[key]:
                    self.key_states[key] = True
                    self.save_screenshot(direction=direction)
                elif not keyboard.is_pressed(key):
                    self.key_states[key] = False

            if show:
                self.show_frame()
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(0.01)  # 降低CPU占用

        cv2.destroyAllWindows()
        print("程序已退出")


# 使用示例
if __name__ == "__main__":
    custom_region = {
        "top": 0,
        "left": 0,
        "width": 2560,
        "height": 1600
    }

    sc = ScreenCapture(region=custom_region, output_path=r"C:\Users\4h55\Pictures\ScreenCaptures")
    sc.run(show=False)