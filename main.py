import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import json

def get_default_dict():
    """默认的内置词库"""
    return {
        'wo': ['我', '窝', '握'],
        'ni': ['你', '泥', '拟'],
        'ta': ['他', '她', '它'],
        'hao': ['好', '号', '豪'],
        'shi': ['是', '时', '十']
    }

def load_chinese_dict(file_path="chinese_dict.json"):
    """从JSON文件加载中文词库"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"词库文件 {file_path} 未找到，使用内置词库")
        return get_default_dict()  # 返回一个默认的小词库
    except json.JSONDecodeError as e:
        print(f"JSON格式错误: {e}")
        print("使用默认词库")
        return get_default_dict()
    except Exception as e:
        print(f"加载词库失败: {e}，使用内置词库")
        return get_default_dict()

# 简化的中文词库
CHINESE_DICT = load_chinese_dict("D:/Jupyter/Pose estimation/KeyBoard/chinese_dict.json")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)

keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "←"],
    ["中/英", "Z", "X", "C", "V", "B", "N", "M", ",", "."],
    ["空格", "确认", "清空"]
]
finalText = ""
pinyin_input = ""
candidates = []
selected_index = 0
input_mode = "chinese"


class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text
        self.state = "normal"
        self.tilted_corners = None
        self.tilted_center = None

    def set_tilted_geometry(self, corners, center):
        self.tilted_corners = corners
        self.tilted_center = center

    def is_point_inside(self, point):
        if point is None or self.tilted_corners is None:
            return False
        return cv2.pointPolygonTest(self.tilted_corners, point, False) >= 0

    def get_color(self):
        colors = {
            "normal": (150, 150, 150, 150),
            "hover": (175, 0, 175, 200),
            "pressing": (0, 100, 255, 255),
            "clicked": (0, 255, 0, 255)
        }
        return colors.get(self.state, (150, 150, 150, 150))


class ClickDetector:
    def __init__(self):
        self.state = "IDLE"
        self.prev_finger_y = None
        self.current_button = None
        self.press_threshold = 15
        self.total_press_distance = 0

    def find_hovered_button(self, finger_pos, buttonList):
        for button in buttonList:
            if button.is_point_inside(finger_pos):
                return button
        return None

    def update(self, finger_pos, hovered_button):
        current_finger_y = finger_pos[1]

        if self.state == "IDLE":
            if hovered_button is not None:
                self.state = "HOVER"
                self.current_button = hovered_button
                self.prev_finger_y = current_finger_y
                return "hover"
            return None

        elif self.state == "HOVER":
            if hovered_button is None or hovered_button != self.current_button:
                self.state = "IDLE"
                self.current_button = None
                return "leave"
            else:
                if self.prev_finger_y is not None:
                    move_distance = current_finger_y - self.prev_finger_y
                    if move_distance > 2:
                        self.total_press_distance += move_distance
                    if self.total_press_distance > self.press_threshold:
                        self.state = "PRESS"
                        return 'pressing'
                self.prev_finger_y = current_finger_y
                return "hover"

        elif self.state == "PRESS":
            if hovered_button is None or hovered_button != self.current_button:
                self.state = "IDLE"
                self.current_button = None
                self.total_press_distance = 0
                return "cancel"
            else:
                if self.prev_finger_y is not None:
                    move_distance = current_finger_y - self.prev_finger_y
                    if move_distance < -5:
                        self.state = "CLICK"
                        return "click"
                    elif move_distance > 0:
                        self.total_press_distance += move_distance
                self.prev_finger_y = current_finger_y
                return "pressing"

        elif self.state == "CLICK":
            clicked_button = self.current_button
            self.state = "IDLE"
            self.current_button = None
            self.prev_finger_y = None
            self.total_press_distance = 0
            return {"action": "complete", "button": clicked_button}

        self.prev_finger_y = current_finger_y
        return None


def put_chinese_text(img, text, position, font_size=30, color=(255, 255, 255)):
    """在OpenCV图像上绘制中文文本"""
    if not text:  # 如果文本为空，直接返回原图像
        return img

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("msyh.ttc", font_size)
        except:
            font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_pinyin_candidates(pinyin_str):
    """根据拼音获取候选词"""
    if pinyin_str in CHINESE_DICT:
        return CHINESE_DICT[pinyin_str]
    else:
        matches = []
        for key, words in CHINESE_DICT.items():
            if key.startswith(pinyin_str):
                matches.extend(words)
        return matches[:5] if matches else []


def handle_button_click(button_text):
    """处理按钮点击事件"""
    global finalText, pinyin_input, candidates, selected_index, input_mode

    if input_mode == "chinese":
        if button_text == "中/英":
            input_mode = "english"
            pinyin_input = ""
            candidates = []
        elif button_text == "←":
            if pinyin_input:
                pinyin_input = pinyin_input[:-1]
                candidates = get_pinyin_candidates(pinyin_input)
                selected_index = 0
            else:
                finalText = finalText[:-1] if finalText else ""
        elif button_text == "空格":
            if candidates:
                finalText += candidates[selected_index]
                pinyin_input = ""
                candidates = []
                selected_index = 0
            else:
                finalText += " "
        elif button_text == "确认":
            if candidates:
                finalText += candidates[selected_index]
                pinyin_input = ""
                candidates = []
                selected_index = 0
        elif button_text == "清空":
            finalText = ""
            pinyin_input = ""
            candidates = []
        elif len(button_text) == 1 and button_text.isalpha():
            pinyin_input += button_text.lower()
            candidates = get_pinyin_candidates(pinyin_input)
            selected_index = 0
    else:
        if button_text == "中/英":
            input_mode = "chinese"
        elif button_text == "←":
            finalText = finalText[:-1] if finalText else ""
        elif button_text == "空格":
            finalText += " "
        elif button_text == "清空":
            finalText = ""
        elif len(button_text) == 1:
            finalText += button_text


def create_keyboard(img, buttonList, tilt_angle=-30):
    """创建倾斜键盘"""
    height, width = img.shape[:2]
    keyboard_w = 1100
    keyboard_h = 450

    start_x = (width - keyboard_w) // 2
    start_y = height - keyboard_h - 50

    src_points = np.float32([
        [start_x, start_y],
        [start_x + keyboard_w, start_y],
        [start_x + keyboard_w, start_y + keyboard_h],
        [start_x, start_y + keyboard_h]
    ])

    tilt_offset = keyboard_h * math.tan(math.radians(tilt_angle))
    dst_points = np.float32([
        [start_x - tilt_offset / 2, start_y],
        [start_x + keyboard_w + tilt_offset / 2, start_y],
        [start_x + keyboard_w - tilt_offset / 2, start_y + keyboard_h],
        [start_x + tilt_offset / 2, start_y + keyboard_h]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    keyboard_canvas = np.zeros((keyboard_h, keyboard_w, 4), dtype=np.uint8)

    # 绘制候选词区域
    if input_mode == "chinese" and (pinyin_input or candidates):
        candidate_h = 60
        cv2.rectangle(keyboard_canvas, (0, 0), (keyboard_w, candidate_h), (100, 100, 100, 200), cv2.FILLED)

        # 使用PIL绘制中文
        keyboard_canvas_rgb = cv2.cvtColor(keyboard_canvas, cv2.COLOR_BGRA2RGBA)
        img_pil = Image.fromarray(keyboard_canvas_rgb)
        draw = ImageDraw.Draw(img_pil)

        try:
            font_small = ImageFont.truetype("simhei.ttf", 20)
            font_smaller = ImageFont.truetype("simhei.ttf", 18)
        except:
            try:
                font_small = ImageFont.truetype("msyh.ttc", 20)
                font_smaller = ImageFont.truetype("msyh.ttc", 18)
            except:
                font_small = ImageFont.load_default()
                font_smaller = ImageFont.load_default()

        # 绘制拼音
        draw.text((10, 15), f"拼音: {pinyin_input}", font=font_small, fill=(255, 255, 255, 255))

        # 绘制候选词
        candidate_text = "候选: "
        for i, word in enumerate(candidates[:5]):
            if i == selected_index:
                candidate_text += f"[{word}] "
            else:
                candidate_text += f"{word} "
        draw.text((10, 40), candidate_text, font=font_smaller, fill=(255, 255, 255, 255))

        keyboard_canvas = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)

    # 绘制按钮
    for button in buttonList:
        canvas_x, canvas_y = button.pos
        w, h = button.size

        if 0 <= canvas_x < keyboard_w and 0 <= canvas_y < keyboard_h:
            button_corners = np.float32([
                [canvas_x, canvas_y],
                [canvas_x + w, canvas_y],
                [canvas_x + w, canvas_y + h],
                [canvas_x, canvas_y + h]
            ]).reshape(-1, 1, 2)

            tilted_corners = cv2.perspectiveTransform(button_corners, matrix)
            center_x = np.mean(tilted_corners[:, 0, 0])
            center_y = np.mean(tilted_corners[:, 0, 1])

            button.set_tilted_geometry(tilted_corners.astype(np.int32),
                                       (int(center_x), int(center_y)))

            color = button.get_color()
            cv2.rectangle(keyboard_canvas, (canvas_x, canvas_y),
                          (canvas_x + w, canvas_y + h), color, cv2.FILLED)

            # 绘制按钮文本（英文用cv2，中文用PIL）
            text_x, text_y = canvas_x + 10, canvas_y + 60
            if button.text in ["中/英", "空格", "确认", "清空"]:
                # 中文按钮用PIL绘制
                keyboard_canvas_rgb = cv2.cvtColor(keyboard_canvas, cv2.COLOR_BGRA2RGBA)
                img_pil = Image.fromarray(keyboard_canvas_rgb)
                draw = ImageDraw.Draw(img_pil)

                try:
                    font_btn = ImageFont.truetype("simhei.ttf", 20)
                except:
                    try:
                        font_btn = ImageFont.truetype("msyh.ttc", 20)
                    except:
                        font_btn = ImageFont.load_default()

                if button.text == "中/英":
                    draw.text((text_x, text_y - 40), button.text, font=font_btn, fill=(255, 255, 255, 255))
                else:
                    draw.text((text_x, text_y - 40), button.text, font=font_btn, fill=(255, 255, 255, 255))

                keyboard_canvas = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
            else:
                # 英文按钮用cv2绘制
                cv2.putText(keyboard_canvas, button.text, (text_x, text_y),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255, 255), 2)

    warped_keyboard = cv2.warpPerspective(keyboard_canvas, matrix, (width, height))
    alpha_channel = warped_keyboard[:, :, 3] / 255.0
    for c in range(3):
        img[:, :, c] = (1 - alpha_channel) * img[:, :, c] + alpha_channel * warped_keyboard[:, :, c]

    return img


# 创建按钮列表
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if key == "空格":
            buttonList.append(Button([100 * j + 50, 100 * i + 100], key, [300, 85]))
        elif key in ["确认", "清空"]:
            buttonList.append(Button([100 * j + 50, 100 * i + 100], key, [200, 85]))
        elif key == "中/英":
            buttonList.append(Button([100 * j + 50, 100 * i + 100], key, [150, 85]))
        else:
            buttonList.append(Button([100 * j + 50, 100 * i + 100], key))

click_detector = ClickDetector()


while True:
    success, img = cap.read()
    if not success:
        continue

    img_original = img.copy()
    hands, img = detector.findHands(img)

    current_hand_pos = None

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        if lmList and len(lmList) > 8:
            current_hand_pos = (lmList[8][0], lmList[8][1])

            hovered_button = click_detector.find_hovered_button(current_hand_pos, buttonList)
            result = click_detector.update(current_hand_pos, hovered_button)

            if result == "hover" and click_detector.current_button:
                click_detector.current_button.state = "hover"

            elif result == "pressing" and click_detector.current_button:
                click_detector.current_button.state = "pressing"

            elif result and isinstance(result, dict) and result["action"] == "complete":
                clicked_button = result["button"]
                clicked_button.state = "clicked"
                handle_button_click(clicked_button.text)
                sleep(0.15)
                clicked_button.state = "normal"

            for button in buttonList:
                if (click_detector.current_button and
                        button != click_detector.current_button and
                        button.state != "normal"):
                    button.state = "normal"

    img = create_keyboard(img, buttonList)

    # 显示输入文本区域
    cv2.rectangle(img, (50, 500), (700, 550), (50, 50, 50), cv2.FILLED)
    img = put_chinese_text(img, finalText, (60, 500), font_size=30, color=(255, 255, 255))
    mode_text = "中文模式" if input_mode == "chinese" else "英文模式"
    img = put_chinese_text(img, mode_text, (50, 450), font_size=25, color=(255, 255, 0))

    cv2.imshow("中文虚拟键盘 - Chinese Virtual Keyboard", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()