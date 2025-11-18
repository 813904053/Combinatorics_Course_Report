# 导入类
from Button import Button
from ClickDetector import ClickDetector
from GestureHandler import GestureHandler
from InputController import InputController
from Keyboard import Keyboard
import utils
import config

# 导入库
import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np

# 导入中文词库
config.CHINESE_DICT = utils.load_chinese_dict("D:/Jupyter/Pose estimation/KeyBoard/chinese_dict.json")

# 显示部分
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)

# 创建按钮列表
buttonList = []
keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ":"],
    ["中/英", "Z", "X", "C", "V", "B", "N", "M", ",", "."],
    ["空格", "确认", "清空"]
]
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j, 100 * i], key))

keyboard = Keyboard(buttonList)

click_detectors = [ClickDetector(i) for i in range(5)]  # 0:食指, 1:中指, 2:无名指, 3:小指


# 帧率
fps_start_time = time.time()
frame_count = 0

# 手势
gesture_handler = GestureHandler()

# 在电脑上输入
input_controller = InputController()

while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    # 每100帧计算一次帧率
    frame_count += 1
    if frame_count % 100 == 0:
        fps = frame_count / (time.time() - fps_start_time)
        frame_count = 0
        fps_start_time = time.time()
    # 在图像上显示FPS（每帧都显示）
    current_fps = frame_count / (time.time() - fps_start_time) if frame_count > 0 else 0
    cv2.putText(img, f"FPS: {current_fps:.1f}", (1000, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    hands, img = detector.findHands(img)

    current_hand_pos = None

    gesture_handler.update()

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        for click_detector in click_detectors:
            utils.HandsUpdate(hands, click_detector, keyboard.buttonList)

        # 手势检测
        if len(lmList) >= 21:
            # 构造凸包点
            list_lms = np.array(lmList, dtype=np.int32)[:, :2]
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
            hull = cv2.convexHull(list_lms[hull_index, :])
            # 绘制凸包
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)

            # 查找外部的点数
            n_fig = -1
            ll = [4, 8, 12, 16, 20]
            up_fingers = []

            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    up_fingers.append(i)

            # print(up_fingers)
            # print(list_lms)
            # print(np.shape(list_lms))
            str_guester = utils.get_str_guester(up_fingers, list_lms)
            utils.check_and_publish_gesture(str_guester, up_fingers, gesture_handler)


            cv2.putText(img, ' %s' % (str_guester), (800, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3,
                        cv2.LINE_AA)

    # 绘制键盘
    img = utils.draw_rectangle_button(img, keyboard)

    # 拼音候选区
    if config.input_mode == "chinese" and config.pinyin_input:
        img = utils.draw_candidates(img)

    # 显示输入文本区域
    cv2.rectangle(img, (0, 0), (700, 50), (50, 50, 50), cv2.FILLED)
    img = utils.put_chinese_text(img, config.finalText, (0, 0), font_size=30, color=(255, 255, 255))
    mode_text = "中文模式" if config.input_mode == "chinese" else "英文模式"
    img = utils.put_chinese_text(img, mode_text, (0, 70), font_size=25, color=(255, 255, 0))
    cv2.imshow("Chinese Virtual Keyboard", img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()