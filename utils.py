import config
from Event import EventBus, GestureEvent
import cv2
import cvzone
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
from time import sleep
import time

# 默认词库（如果没有json文件）
def get_default_dict():
    """默认的内置词库"""
    return {
        'wo': ['我', '窝', '握'],
        'ni': ['你', '泥', '拟'],
        'ta': ['他', '她', '它'],
        'hao': ['好', '号', '豪'],
        'shi': ['是', '时', '十']
    }


# 从JSON文件加载中文词库
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


# 用PIL绘制，已舍弃
def draw_keyboard_pil(img, keyboard):
    x, y = keyboard.pos
    w, h = keyboard.w, keyboard.h

    # 第一步：将OpenCV图像转换为PIL图像

    # 将图像从 BGR 颜色空间转换为 RGB 颜色空间
    # OpenCV 默认使用 BGR 格式（蓝-绿-红）
    # PIL 使用 RGB 格式（红-绿-蓝）
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 将 NumPy 数组转换为 PIL Image 对象
    # OpenCV 图像本质是 NumPy 数组
    # PIL 需要自己的 Image 对象来进行绘图操作
    draw = ImageDraw.Draw(img_pil, 'RGBA')  # 使用RGBA模式

    # 绘制半透明矩形
    draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, 128))  # 128是透明度

    # 转换回OpenCV格式
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img


# 绘制矩形（开发用）
def draw_keyboard(img, keyboard, alpha=0.3, angle=0):
    x, y = keyboard.pos
    w, h = keyboard.w, keyboard.h

    # 使用深色作为键盘背景
    color = (50, 50, 50)  # 深灰色

    # 提取ROI区域
    roi = img[y:y+h, x:x+w]

    # 键盘颜色
    color_layer = np.full_like(roi, color)

    # 将ROI处，原始图像和键盘图像混合
    blended = cv2.addWeighted(roi, 1 - alpha, color_layer, alpha, 0)
    img[y:y+h, x:x+w] = blended

    # 添加键盘边框
    cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 100), 2)

    return img


# 绘制矩形（开发用）
def draw_rectangle(img, points, alpha=0.3, angle=0):
    x, y = points[0].astype(int)
    w, h = int(points[2][0]-points[0][0]), int(points[2][1] - points[0][1])

    # 使用深色作为键盘背景
    color = (50, 50, 50)  # 深灰色

    # 提取ROI区域
    roi = img[y:y+h, x:x+w]

    # 键盘颜色
    color_layer = np.full_like(roi, color)

    # 将ROI处，原始图像和键盘图像混合
    blended = cv2.addWeighted(roi, 1 - alpha, color_layer, alpha, 0)
    img[y:y+h, x:x+w] = blended

    # 添加键盘边框
    cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 100), 2)

    return img


# 绘制旋转矩形（开发用）
def draw_rectangle_rotate(img, points, alpha=0.3, angle=0):
    points = points.astype(np.float32)

    x, y = points[0].astype(int)
    w = int(points[2][0] - points[0][0])
    h = int(points[2][1] - points[0][1])

    # 计算透视偏移
    offset = h * math.tan(math.radians(angle))

    # 目标点
    dst_points = np.float32([
        [x - offset / 2, y],
        [x + w + offset / 2, y],
        [x + w - offset / 2, y + h],
        [x + offset / 2, y + h]
    ])

    # 直接绘制填充和边框
    color = (50, 50, 50)

    # 绘制填充
    overlay = img.copy()
    # 填充键盘底部
    cv2.fillConvexPoly(overlay,  dst_points.astype(np.int32), color)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 绘制边框
    cv2.polylines(img, [dst_points], True, (100, 100, 100), 2)

    return img


# 绘制键盘和按键
def draw_rectangle_button(img, keyboard, alpha=0.3, angle=-30):
    points = keyboard.points

    x, y = points[0].astype(int)
    w = int(points[2][0] - points[0][0])
    h = int(points[2][1] - points[0][1])

    # 计算透视偏移
    offset = h * math.tan(math.radians(angle))

    # 目标点
    dst_points = np.float32([
        [x - offset / 2, y],
        [x + w + offset / 2, y],
        [x + w - offset / 2, y + h],
        [x + offset / 2, y + h]
    ])

    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(points, dst_points)

    # 直接绘制填充和边框
    color = (50, 50, 50)

    # 绘制填充
    overlay = img.copy()
    # 填充键盘底部
    cv2.fillConvexPoly(overlay, dst_points.astype(np.int32), color)

    for button in keyboard.buttonList:
        button_x, button_y = button.pos
        button_w, button_h = button.size
        button_points = button.points.astype(np.float32)

        # 应用相同的透视变换到按键
        button.tilted_corners = cv2.perspectiveTransform(
            button_points.reshape(1, -1, 2), matrix
        )[0].astype(np.int32)

        # 绘制按钮
        cv2.fillConvexPoly(overlay, button.tilted_corners, (*button.color, 200))  # 150是透明度

        # 绘制按钮文字
        if button.text:
            # 计算按钮中心点
            center_x = np.mean(button.tilted_corners[:, 0])
            center_y = np.mean(button.tilted_corners[:, 1])

            # 使用PIL绘制中文文本
            if button.text in ["中/英", "空格", "确认", "清空"]:
                # 使用PIL绘制中文
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(overlay_rgb)
                draw = ImageDraw.Draw(img_pil)

                try:
                    font = ImageFont.truetype("simhei.ttf", 20)
                except:
                    try:
                        font = ImageFont.truetype("msyh.ttc", 20)
                    except:
                        font = ImageFont.load_default()

                # 获取文本尺寸（PIL方式）
                bbox = draw.textbbox((0, 0), button.text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # 计算居中位置
                text_x = int(center_x - text_width / 2)
                text_y = int(center_y - text_height / 2)

                # 绘制文本
                draw.text((text_x, text_y), button.text, font=font, fill=(255, 255, 255))

                # 转换回OpenCV格式
                overlay_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                overlay[:, :, :] = overlay_bgr[:, :, :]
            else:
                # 英文按钮 - 使用OpenCV
                text_size = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = int(center_x - text_size[0] / 2)
                text_y = int(center_y + text_size[1] / 2)
                cv2.putText(overlay, button.text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 绘制边框
    cv2.polylines(img, [dst_points.astype(np.int32)], True, (100, 100, 100), 2)

    return img


# 绘制候选区
def draw_candidates(img):
    candidate_h = 60
    cv2.rectangle(img, (0, 100), (300, 100+candidate_h), (100, 100, 100, 200), cv2.FILLED)

    # 使用PIL绘制中文
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    img_pil = Image.fromarray(img_rgb)
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
    draw.text((10, 105), f"拼音: {config.pinyin_input}", font=font_small, fill=(255, 255, 255, 255))

    # 绘制候选词
    candidate_text = "候选: "
    for i, word in enumerate(config.candidates[:5]):
        if i == config.selected_index:
            candidate_text += f"[{word}] "
        else:
            candidate_text += f"{word} "
    draw.text((10, 125), candidate_text, font=font_smaller, fill=(255, 255, 255, 255))

    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGBA2BGRA)
    return img


# 在OpenCV图像上绘制中文文本
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


# 根据拼音获取候选词
def get_pinyin_candidates(pinyin_str):
    """根据拼音获取候选词"""
    if pinyin_str in config.CHINESE_DICT:
        return config.CHINESE_DICT[pinyin_str]
    else:
        matches = []
        for key, words in config.CHINESE_DICT.items():
            if key.startswith(pinyin_str):
                matches.extend(words)
        return matches[:5] if matches else []


# 处理按钮点击事件
def handle_button_click(button_text):
    """处理按钮点击事件"""
    if config.input_mode == "chinese":
        if button_text == "中/英":
            config.input_mode = "english"
            config.pinyin_input = ""
            config.candidates = []
        elif button_text == "←":
            if config.pinyin_input:
                config.pinyin_input = config.pinyin_input[:-1]
                config.candidates = get_pinyin_candidates(config.pinyin_input)
                config.selected_index = 0
            else:
                config.finalText = config.finalText[:-1] if config.finalText else ""
        elif button_text == "空格":
            if config.candidates:
                config.finalText += config.candidates[config.selected_index]
                config.pinyin_input = ""
                config.candidates = []
                config.selected_index = 0
            else:
                config.finalText += " "
        elif button_text == "确认":
            if config.candidates:
                config.finalText += config.candidates[config.selected_index]
                config.pinyin_input = ""
                config.candidates = []
                config.selected_index = 0
        elif button_text == "清空":
            config.finalText = ""
            config.pinyin_input = ""
            config.candidates = []
        elif len(button_text) == 1 and button_text.isalpha():
            config.pinyin_input += button_text.lower()
            config.candidates = get_pinyin_candidates(config.pinyin_input)
            config.selected_index = 0
    else:
        if button_text == "中/英":
            config.input_mode = "chinese"
        elif button_text == "←":
            config.finalText = config.finalText[:-1] if config.finalText else ""
        elif button_text == "空格":
            config.finalText += " "
        elif button_text == "清空":
            config.finalText = ""
        elif len(button_text) == 1:
            config.finalText += button_text


def HandsUpdate(hands, click_detector, buttonList):
    hand = hands[0]
    lmList = hand["lmList"]

    joint = config.id2joint[click_detector.id]

    if lmList and len(lmList) > joint:
        current_hand_pos = (lmList[joint][0], lmList[joint][1])

        hovered_button = click_detector.find_hovered_button(current_hand_pos, buttonList)
        click_detector.update(current_hand_pos, hovered_button)


def get_angle(v1,v2):
    angle = np.dot(v1,v2)/(np.sqrt(np.sum(v1*v1))*np.sqrt(np.sum(v2*v2)))
    angle = np.arccos(angle)/3.14*180
    return angle


def get_str_guester(up_fingers, list_lms):
    if len(up_fingers) == 1 and up_fingers[0] == 8:

        v1 = list_lms[6] - list_lms[7]
        v2 = list_lms[8] - list_lms[7]

        angle = get_angle(v1, v2)

        if angle < 160:
            str_guester = "9"
        else:
            str_guester = "1"

    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        str_guester = "thumbs_up"

    elif len(up_fingers) == 1 and up_fingers[0] == 20:
        str_guester = "Bad"

    elif len(up_fingers) == 1 and up_fingers[0] == 12:
        str_guester = "FXXX"

    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        str_guester = "2"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:
        str_guester = "6"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        str_guester = "8"

    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        str_guester = "3"

    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 12:

        dis_8_12 = list_lms[8, :] - list_lms[12, :]
        dis_8_12 = np.sqrt(np.dot(dis_8_12, dis_8_12))

        dis_4_12 = list_lms[4, :] - list_lms[12, :]
        dis_4_12 = np.sqrt(np.dot(dis_4_12, dis_4_12))

        if dis_4_12 / (dis_8_12 + 1) < 3:
            str_guester = "7"

        elif dis_4_12 / (dis_8_12 + 1) > 5:
            str_guester = "Gun"
        else:
            str_guester = "7"

    elif len(up_fingers) == 3 and up_fingers[0] == 4 and up_fingers[1] == 8 and up_fingers[2] == 20:
        str_guester = "ROCK"

    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[
        3] == 20:
        str_guester = "4"

    elif len(up_fingers) == 5:
        str_guester = "5"

    elif len(up_fingers) == 0:
        str_guester = "fist"

    else:
        str_guester = " "

    return str_guester


def check_and_publish_gesture(str_guester, up_fingers, gesture_handler):
    if gesture_handler.gesture_cooldown > 0:
        gesture_handler.gesture_cooldown -= 1
        return

    if str_guester != gesture_handler.last_gesture and str_guester in ["thumbs_up", "fist", "open_hand", "victory", "ok"]:
        # 映射手势名称到事件类型
        gesture_event_map = {
            "thumbs_up": GestureEvent.THUMBS_UP,
            "fist": GestureEvent.FIST,  # 握拳
            "5": GestureEvent.OPEN_HAND,  # 手掌张开
            "2": GestureEvent.VICTORY,  # 比耶
            "8": GestureEvent.OK  # OK手势
        }

        event_type = gesture_event_map.get(str_guester)
        if event_type:
            EventBus.publish(event_type, {
                "gesture_type": str_guester,
                "up_fingers": up_fingers,
                "timestamp": time.time()
            })
            gesture_handler.last_gesture = str_guester
            gesture_handler.gesture_cooldown = 10  # 冷却时间