# 导入类
from Button import Button
from ClickDetector import ClickDetector
from Keyboard import Keyboard
import utils
import config

# 导入库
import cv2
from cvzone.HandTrackingModule import HandDetector

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

click_detector = ClickDetector()

while True:
    success, img = cap.read()
    if not success:
        continue

    img_original = img.copy()
    hands, img = detector.findHands(img)

    current_hand_pos = None

    if hands:
        utils.HandsUpdate(hands, click_detector, keyboard.buttonList)

    # 绘制键盘
    img = utils.draw_rectangle_button(img, keyboard)

    # 拼音候选区
    if config.input_mode == "chinese" and (config.pinyin_input or config.candidates):
        img = utils.draw_candidates(img)

    # 显示输入文本区域
    cv2.rectangle(img, (0, 0), (700, 50), (50, 50, 50), cv2.FILLED)
    img = utils.put_chinese_text(img, config.finalText, (0, 0), font_size=30, color=(255, 255, 255))
    mode_text = "中文模式" if config.input_mode == "chinese" else "英文模式"
    img = utils.put_chinese_text(img, mode_text, (0, 70), font_size=25, color=(255, 255, 0))

    cv2.imshow("Chinese Virtual Keyboard", img)

    if click_detector.current_button:
        cv2.putText(img, click_detector.current_button.state, (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()