import cv2
import os
import time
import sys
import traceback
from functools import partial
from game import game

class RockPaperScissorsGame:
    def __init__(self):
        self.SAVE_DIR = "captured_images"
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        self.FIXED_FILENAME = "my_gesture.jpg"
        self.image_path = os.path.join(self.SAVE_DIR, self.FIXED_FILENAME)

        self.GESTURE_IMAGES = {
            "rock": os.path.join("cartoon", "rock.jpg"),
            "paper": os.path.join("cartoon", "paper.jpg"),
            "scissors": os.path.join("cartoon", "scissors.jpg")
        }

        for img_path in self.GESTURE_IMAGES.values():
            dir_name = os.path.dirname(img_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        self.STATE_CAPTURE = 0
        self.STATE_COUNTDOWN = 1
        self.STATE_RESULT = 2

        self.running = True
        self.current_frame = None
        self.raw_frame = None
        self.window_name = "Rock Paper Scissors"
        self.current_state = self.STATE_CAPTURE
        self.countdown_start = 0

        self.player_gesture = ""
        self.computer_gesture = ""
        self.game_result = ""
        self.my_score = 0
        self.computer_score = 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        mouse_callback = partial(self.mouse_callback)
        cv2.setMouseCallback(self.window_name, mouse_callback)


        self.cap = self.init_camera()

    def init_camera(self):
        for api in [cv2.CAP_ANY, cv2.CAP_DSHOW] if sys.platform.startswith('win') else [cv2.CAP_ANY]:
            for index in [0, 1, 2]:
                try:
                    cap = cv2.VideoCapture(index, api)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap
                except Exception as e:
                    print(f"尝试初始化摄像头 {index} 失败: {e}")
                    continue

        return None

    def draw_capture_buttons(self, frame):
        frame_height, frame_width = frame.shape[:2]

        # 中心Capture按钮
        cap_btn_width, cap_btn_height = 200, 80
        cap_x1 = (frame_width - cap_btn_width) // 2
        cap_y1 = (frame_height - cap_btn_height) // 2
        cap_x2 = cap_x1 + cap_btn_width
        cap_y2 = cap_y1 + cap_btn_height
        cv2.rectangle(frame, (cap_x1, cap_y1), (cap_x2, cap_y2), (0, 255, 0), -1)
        cv2.putText(frame, "Capture", (cap_x1 + 40, cap_y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        quit_btn_width, quit_btn_height = 120, 60
        quit_x1 = frame_width - quit_btn_width - 20
        quit_y1 = frame_height - quit_btn_height - 20
        quit_x2 = quit_x1 + quit_btn_width
        quit_y2 = quit_y1 + quit_btn_height
        cv2.rectangle(frame, (quit_x1, quit_y1), (quit_x2, quit_y2), (0, 0, 255), -1)
        cv2.putText(frame, "Quit", (quit_x1 + 25, quit_y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return {"Capture": (cap_x1, cap_y1, cap_x2, cap_y2), "Quit": (quit_x1, quit_y1, quit_x2, quit_y2)}

    def draw_continue_button(self, frame):
        frame_height, frame_width = frame.shape[:2]

        cont_btn_width, cont_btn_height = 200, 80
        cont_x1 = (frame_width - cont_btn_width) // 2
        cont_y1 = frame_height - cont_btn_height - 30
        cont_x2 = cont_x1 + cont_btn_width
        cont_y2 = cont_y1 + cont_btn_height
        cv2.rectangle(frame, (cont_x1, cont_y1), (cont_x2, cont_y2), (255, 255, 0), -1)
        cv2.putText(frame, "Continue", (cont_x1 + 20, cont_y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        quit_x1 = frame_width - 120 - 20
        quit_y1 = frame_height - 60 - 20
        quit_x2 = quit_x1 + 120
        quit_y2 = quit_y1 + 60
        return {"Continue": (cont_x1, cont_y1, cont_x2, cont_y2),
                "Quit": (quit_x1, quit_y1, quit_x2, quit_y2)}

    def draw_scores(self, frame):
        frame_width = frame.shape[1]

        cv2.putText(frame, "Rock Paper Scissors",
                    (frame_width // 2 - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        score_text = f"Player's score: {self.my_score} vs Computer's score: {self.computer_score}"
        cv2.putText(frame, score_text,
                    (frame_width // 2 - 220, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_countdown(self, frame):
        frame_height, frame_width = frame.shape[:2]
        center = (frame_width // 2, frame_height // 2)
        radius = 100

        cv2.circle(frame, center, radius, (0, 0, 0), -1)
        cv2.circle(frame, center, radius, (255, 255, 255), 5)

        elapsed = time.time() - self.countdown_start
        remaining = max(0, 3 - int(elapsed))

        text = str(remaining)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 5)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

        return remaining

    def draw_results(self, frame):
        frame_height, frame_width = frame.shape[:2]
        gesture_size = (200, 200)

        if self.player_gesture in self.GESTURE_IMAGES:
            try:
                img_path = self.GESTURE_IMAGES[self.player_gesture]
                if os.path.exists(img_path):
                    player_img = cv2.imread(img_path)
                    if player_img is not None:
                        player_img = cv2.resize(player_img, gesture_size)
                        player_x = 50
                        player_y = (frame_height - gesture_size[1]) // 2
                        if player_y + gesture_size[1] <= frame_height and player_x + gesture_size[0] <= frame_width:
                            frame[player_y:player_y + gesture_size[1], player_x:player_x + gesture_size[0]] = player_img
                    else:
                        cv2.putText(frame, "Image corrupted", (50, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"No {self.player_gesture} image", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            except Exception as e:
                print(f"玩家手势绘制错误: {e}")
                cv2.putText(frame, "Load failed", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 绘制电脑手势（右侧）- 移除"Computer's Gesture"标签
        if self.computer_gesture in self.GESTURE_IMAGES:
            try:
                img_path = self.GESTURE_IMAGES[self.computer_gesture]
                if os.path.exists(img_path):
                    comp_img = cv2.imread(img_path)
                    if comp_img is not None:
                        comp_img = cv2.resize(comp_img, gesture_size)
                        comp_x = frame_width - gesture_size[0] - 50
                        comp_y = (frame_height - gesture_size[1]) // 2
                        if comp_y + gesture_size[1] <= frame_height and comp_x + gesture_size[0] <= frame_width:
                            frame[comp_y:comp_y + gesture_size[1], comp_x:comp_x + gesture_size[0]] = comp_img
                    else:
                        cv2.putText(frame, "Image corrupted", (frame_width - 250, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"No {self.computer_gesture} image", (frame_width - 250, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            except Exception as e:
                print(f"电脑手势绘制错误: {e}")
                cv2.putText(frame, "Load failed", (frame_width - 250, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if not self.game_result:
            display_result = "Processing..."
            result_color = (255, 255, 255)
        else:
            display_result = self.game_result
            if "Win" in display_result:
                result_color = (0, 255, 0)  # 绿色
            elif "Lose" in display_result:
                result_color = (0, 0, 255)  # 红色
            elif "Draw" in display_result:
                result_color = (0, 255, 255)  # 黄色
            else:
                result_color = (0, 0, 255)  # 红色

        text_size = cv2.getTextSize(display_result, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = max(0, min(frame_width - text_size[0], frame_width // 2 - text_size[0] // 2))
        text_y = frame_height // 2
        cv2.putText(frame, display_result,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, result_color, 3)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.raw_frame is None:
                return

            if self.current_state == self.STATE_CAPTURE:
                buttons = self.draw_capture_buttons(self.current_frame)
                # 点击Quit按钮
                if buttons["Quit"][0] <= x <= buttons["Quit"][2] and buttons["Quit"][1] <= y <= buttons["Quit"][3]:
                    self.running = False
                # 点击Capture按钮
                elif buttons["Capture"][0] <= x <= buttons["Capture"][2] and buttons["Capture"][1] <= y <= \
                        buttons["Capture"][3]:
                    try:
                        cv2.imwrite(self.image_path, self.raw_frame)
                        print(f"截图已保存: {self.image_path}")
                        self.current_state = self.STATE_COUNTDOWN
                        self.countdown_start = time.time()
                    except Exception as e:
                        print(f"保存图片失败: {e}")
                        self.game_result = "Failed to save image"
                        self.current_state = self.STATE_RESULT

            elif self.current_state == self.STATE_RESULT:
                buttons = self.draw_continue_button(self.current_frame)
                # 点击Quit按钮
                if buttons["Quit"][0] <= x <= buttons["Quit"][2] and buttons["Quit"][1] <= y <= buttons["Quit"][3]:
                    self.running = False
                # 点击Continue按钮
                elif buttons["Continue"][0] <= x <= buttons["Continue"][2] and buttons["Continue"][1] <= y <= \
                        buttons["Continue"][3]:
                    self.player_gesture = ""
                    self.computer_gesture = ""
                    self.game_result = ""
                    self.current_state = self.STATE_CAPTURE

    def run(self):
        if not self.cap or not self.cap.isOpened():
            print("错误: 无法访问摄像头")
            print("请确保摄像头已连接并授予权限")
            return

        print("操作说明: 点击Capture拍照 / 按c键拍照 | 点击Quit退出 / 按q键退出")

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("警告: 无法捕获图像，尝试重新获取...")
                    time.sleep(0.1)
                    continue

                frame = cv2.flip(frame, 1)

                self.raw_frame = frame.copy()
                self.current_frame = frame.copy()

                self.draw_scores(self.current_frame)

                if self.current_state == self.STATE_CAPTURE:
                    self.draw_capture_buttons(self.current_frame)

                elif self.current_state == self.STATE_COUNTDOWN:
                    remaining = self.draw_countdown(self.current_frame)
                    if (time.time() - self.countdown_start) >= 3:
                        try:
                            if not os.path.exists(self.image_path):
                                self.game_result = f"Error: Image not found"
                                self.current_state = self.STATE_RESULT
                                continue

                            try:
                                test_img = cv2.imread(self.image_path)
                                if test_img is None:
                                    self.game_result = "Error: Corrupted image"
                                    self.current_state = self.STATE_RESULT
                                    continue
                            except Exception as e:
                                self.game_result = f"Error: Invalid image - {str(e)[:20]}"
                                self.current_state = self.STATE_RESULT
                                continue

                            self.player_gesture, self.computer_gesture, self.game_result = game(self.image_path)
                            print(f"\n识别结果:")
                            print(f"玩家手势: {self.player_gesture}")
                            print(f"电脑手势: {self.computer_gesture}")
                            print(f"游戏结果: {self.game_result}")

                            if "Win" in self.game_result:
                                self.my_score += 1
                            elif "Lose" in self.game_result:
                                self.computer_score += 1
                        except Exception as e:
                            print(f"游戏逻辑错误: {e}")
                            print(traceback.format_exc())
                            self.game_result = "Game logic error"
                        self.current_state = self.STATE_RESULT

                elif self.current_state == self.STATE_RESULT:
                    self.draw_results(self.current_frame)
                    self.draw_continue_button(self.current_frame)

                cv2.imshow(self.window_name, self.current_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('c') and self.current_state == self.STATE_CAPTURE:
                    try:
                        cv2.imwrite(self.image_path, self.raw_frame)
                        print(f"按c键保存截图: {self.image_path}")
                        self.current_state = self.STATE_COUNTDOWN
                        self.countdown_start = time.time()
                    except Exception as e:
                        print(f"保存图片失败: {e}")
                        self.game_result = "Failed to save image"
                        self.current_state = self.STATE_RESULT

        except Exception as e:
            print(f"游戏主循环出错: {e}")
            print(traceback.format_exc())
        finally:
            # 释放资源
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("游戏已结束")


if __name__ == "__main__":
    import matplotlib
    matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    game_app = RockPaperScissorsGame()
    game_app.run()