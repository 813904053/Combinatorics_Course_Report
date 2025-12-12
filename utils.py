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
from Kalmanfilter import MultiHandKalmanFilter

hand_kalman_filter = MultiHandKalmanFilter(process_variance=0.01, measurement_variance=0.1)

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


# 绘制键盘和按键
def draw_rectangle_button(img, keyboard, alpha=0.6, angle=-30):
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
            if button.text in ["中/英", "空格", "确认", "清空", "删除"]:
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

    # 混合
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


# 点击检测
def HandsUpdate(hands, click_detector, buttonList, input_controller):
    hand = hands[0]
    lmList = hand["lmList"]

    joint = config.id2joint[click_detector.id]

    if lmList and len(lmList) > joint:
        current_hand_pos = (lmList[joint][0], lmList[joint][1])

        hovered_button = click_detector.find_hovered_button(current_hand_pos, buttonList)
        click_detector.update(current_hand_pos, hovered_button, input_controller)


def apply_kalman_filter_to_hands(hands):
    """
    对检测到的手部关键点应用卡尔曼滤波

    Args:
        hands: 手部检测结果列表

    Returns:
        应用滤波后的手部列表
    """
    return hand_kalman_filter.update(hands)

# 手势识别
def check_and_publish_gesture(guester, gesture_handler):
    if gesture_handler.gesture_cooldown > 0:
        gesture_handler.gesture_cooldown -= 1
        return
    if gesture_handler.confidence < config.confidence_threshold:
        return
    # 映射手势名称到事件类型
    gesture_event_map = {
        "1": GestureEvent.FIST,
        "2": GestureEvent.OPEN_HAND,
        "3": GestureEvent.THUMBS_UP,
        "4": GestureEvent.THUMBS_LEFT,
        "5": GestureEvent.PALM,
        "6": GestureEvent.SPIDER
    }
    if guester == "1" and gesture_handler.check_gesture_duration(guester):
        print("ready")
        gesture_handler.allow_recognition = True
        config.can_type_in = False

    if guester == "2" and gesture_handler.check_gesture_duration(guester):
        print("cancel")
        gesture_handler.allow_recognition = False
        config.can_type_in = True

    if gesture_handler.allow_recognition:
        if guester == "1":
            return
        event_type = gesture_event_map.get(guester)
        if event_type:
            EventBus.publish(event_type, {
                "gesture_type": guester,
                "timestamp": time.time()
            })
            gesture_handler.last_gesture = guester

