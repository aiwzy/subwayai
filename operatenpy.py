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
        self.auto_interval = 1.0  # 自动截图间隔（秒）

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
        """捕获当前屏幕帧"""
        with mss.mss() as sct:
            img = sct.grab(self.region)
            self.frame = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
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

    def show_frame(self, window_name="Screen Capture"):
        """实时显示当前帧"""
        if self.frame is not None:
            fps = 1.0 / (time.time() - self.last_time)
            cv2.putText(self.frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示操作提示
            cv2.putText(self.frame, "自动保存: 每秒1张 (auto目录)", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(self.frame, "按键保存: W=上 S=下 A=左 D=右", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(window_name, self.frame)
            self.last_time = time.time()

    def run(self, show=True):
        """启动双模式截图（自动+按键）"""
        print("截图程序启动 - 双模式运行:")
        print("  自动模式: 每秒保存1张到 auto 目录")
        print("  按键模式: W/S/A/D 分别保存到 up/down/left/right 目录")
        print("按 ESC 退出程序")

        while not keyboard.is_pressed('esc'):
            self.capture_frame()

            # 处理自动保存（每秒1次）
            if time.time() - self.last_auto_save >= self.auto_interval:
                self.save_screenshot(direction='auto')
                self.last_auto_save = time.time()

            # 处理按键保存
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

    sc = ScreenCapture(region=custom_region, output_path="five_directories/")
    sc.run(show=False)