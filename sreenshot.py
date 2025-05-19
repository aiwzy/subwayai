import mss
import cv2
import numpy as np
import time
import keyboard
import os
from datetime import datetime


class ScreenCapture:
    def __init__(self, region=None, output_path="screenshots/"):
        """初始化屏幕截图工具"""
        self.region = region or self.get_fullscreen_region()
        self.output_path = output_path
        self.frame = None
        self.last_time = time.time()
        self.auto_interval = 0.05  # 自动截图间隔（0.05秒）
        self.auto_screenshots = []  # 自动截图缓存（最多保存4张）
        self.max_cache_size = 4

        # 定义保存目录（自动截图不保存文件，仅方向目录）
        self.directories = {
            'up': os.path.join(output_path, 'up'),
            'down': os.path.join(output_path, 'down'),
            'left': os.path.join(output_path, 'left'),
            'right': os.path.join(output_path, 'right')
        }

        # 创建所有目录
        for dir_path in self.directories.values():
            os.makedirs(dir_path, exist_ok=True)

        # 按键映射
        self.key_mapping = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
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
            self.frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2GRAY)
            return self.frame

    def cache_auto_screenshot(self):
        """仅缓存自动截图，不保存到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.auto_screenshots.append({
            'frame': self.frame.copy(),  # 保存副本
            'timestamp': timestamp
        })
        # 保持缓存大小
        if len(self.auto_screenshots) > self.max_cache_size:
            self.auto_screenshots.pop(0)
        return timestamp

    def save_direction_screenshot(self, direction):
        """保存按键触发的方向截图（JPG格式）"""
        if direction not in self.directories:
            print(f"无效方向: {direction}")
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dir_path = self.directories[direction]
        filename = os.path.join(dir_path, f"{direction}_{timestamp}.jpg")
        cv2.imwrite(filename, self.frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"方向截图保存: {filename}")
        return filename

    def create_composite_image(self, direction):
        """创建合成图（保存为PNG图像和原始张量NPY）"""
        if len(self.auto_screenshots) < 4:
            print("缓存不足，至少需要4张自动截图")
            return None, None, None

        # 提取缓存中的截图（第1、3、4张，索引从0开始）
        img1 = self.auto_screenshots[-4]['frame']  # 最早的缓存
        img2 = self.auto_screenshots[-2]['frame']  # 中间的缓存
        img3 = self.frame  # 当前帧

        h, w = img3.shape[:2]
        img1 = cv2.resize(img1, (w, h))
        img2 = cv2.resize(img2, (w, h))

        # 生成原始张量 [3, H, W]
        composite_tensor = np.stack([img1, img2, img3], axis=0)  # 维度: [3, H, W]

        # 转换为OpenCV可保存的格式 [H, W, 3]
        composite_img = np.transpose(composite_tensor, (1, 2, 0))  # 转置为 [H, W, 3]
        #composite_img = cv2.cvtColor(composite_img, cv2.COLOR_GRAY2BGR)  # 灰度转三通道

        # 保存PNG图像（供查看）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dir_path = self.directories[direction]
        img_filename = os.path.join(dir_path, f"composite_{direction}_{timestamp}.png")
        cv2.imwrite(img_filename, composite_img)

        # 保存原始张量（供模型使用）
        tensor_filename = os.path.join(dir_path, f"tensor_{direction}_{timestamp}.npy")
        np.save(tensor_filename, composite_tensor)

        print(f"合成图保存: {img_filename}")
        print(f"原始张量保存: {tensor_filename}")
        return composite_tensor, img_filename, tensor_filename
    def show_frame(self, window_name="Screen Capture"):
        """实时显示当前帧（带操作提示）"""
        if self.frame is not None:
            fps = 1.0 / (time.time() - self.last_time)
            cv2.putText(self.frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(self.frame, "自动缓存: 每0.05秒 (仅内存)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(self.frame, "按键操作: W/S/A/D 生成合成图", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, self.frame)
            self.last_time = time.time()

    def run(self, show=True):
        """启动程序主循环"""
        print("程序启动 - 功能说明:")
        print("  自动模式: 每0.05秒缓存截图（内存中，最多4张）")
        print("  按键模式: W/S/A/D 生成合成图（保存PNG和原始张量）")
        print("按 ESC 退出程序")

        while not keyboard.is_pressed('esc'):
            self.capture_frame()

            # 处理自动缓存
            if time.time() - self.last_auto_save >= self.auto_interval:
                self.cache_auto_screenshot()
                self.last_auto_save = time.time()

            # 处理按键事件
            for key, direction in self.key_mapping.items():
                if keyboard.is_pressed(key) and not self.key_states[key]:
                    self.key_states[key] = True
                    self.save_direction_screenshot(direction)  # 保存当前帧截图
                    tensor, img_path, tensor_path = self.create_composite_image(direction)  # 生成合成图
                    if show and tensor is not None:
                        print(f"合成图维度: {tensor.shape}")  # 输出: (3, H, W)

            if show:
                self.show_frame()
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            time.sleep(0.01)  # 降低CPU占用

        cv2.destroyAllWindows()
        print("程序退出")

    @staticmethod
    def load_composite_tensor(file_path):
        """加载原始张量（恢复 [3, H, W] 维度）"""
        try:
            tensor = np.load(file_path)
            print(f"成功加载张量，维度: {tensor.shape}")
            return tensor
        except Exception as e:
            print(f"加载失败: {e}")
            return None


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 自定义截图区域（示例：全屏）
    custom_region = {
        "top": 0,
        "left": 0,
        "width": 1920,  # 屏幕宽度
        "height": 1080  # 屏幕高度
    }

    # 创建实例（指定输出路径）
    sc = ScreenCapture(
        region=custom_region,
        output_path=r"C:\Users\4h55\Pictures\ScreenCaptures"
    )

    # 启动程序（show=True 显示实时画面）
    sc.run(show=False)

    # ================= 加载示例 =================
    # 假设已知张量文件路径
    # tensor_path = r"C:\Users\YourUsername\Pictures\ScreenCaptures\left\tensor_left_20250519_123456789.npy"
    # loaded_tensor = ScreenCapture.load_composite_tensor(tensor_path)
    # if loaded_tensor is not None:
    #     print("用于模型输入的张量形状:", loaded_tensor.shape)