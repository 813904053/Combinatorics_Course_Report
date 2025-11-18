from Event import EventBus, ButtonEvent
import utils
import cv2
import numpy as np

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text
        self.color = (150, 150, 150)
        self.tilted_corners = None
        self.tilted_center = None

        self.is_func = self.text in ("中/英", "空格", "确认", "清空")

        self.points = np.float32([
            [self.pos[0], self.pos[1]],
            [self.pos[0] + self.size[0], self.pos[1]],
            [self.pos[0] + self.size[0], self.pos[1] + self.size[1]],
            [self.pos[0], self.pos[1] + self.size[1]]
        ])

        # 订阅事件
        EventBus.subscribe(ButtonEvent.IDLE_HOVER, self.idle_hover_call)
        EventBus.subscribe(ButtonEvent.HOVER_IDLE, self.hover_idle_call)
        EventBus.subscribe(ButtonEvent.HOVER_PRESS, self.hover_press_call)
        EventBus.subscribe(ButtonEvent.PRESS_IDLE, self.press_idle_call)
        EventBus.subscribe(ButtonEvent.PRESS_CLICK, self.press_click_call)

    # 校正位置
    def pos_adjust(self, keyboard_pos, offset):
        self.pos = [self.pos[0] + keyboard_pos[0] + offset,
                      self.pos[1] + keyboard_pos[1] + offset]
        self.points = np.float32([
            [self.pos[0], self.pos[1]],
            [self.pos[0] + self.size[0], self.pos[1]],
            [self.pos[0] + self.size[0], self.pos[1] + self.size[1]],
            [self.pos[0], self.pos[1] + self.size[1]]
        ])

    def is_point_inside(self, point):
        if point is None or self.tilted_corners is None:
            return False
        return cv2.pointPolygonTest(self.tilted_corners, point, False) >= 0

    def idle_hover_call(self, data):
        if data["button"] == self:
            self.color = (0, 0, 0)

    def hover_idle_call(self, data):
        if data["button"] == self:
            self.color = (150, 150, 150)

    def hover_press_call(self, data):
        if data["button"] == self:
            self.color = (0, 100, 255)

    def press_idle_call(self, data):
        if data["button"] == self:
            self.color = (150, 150, 150)

    def press_click_call(self, data):
        if data["button"] == self:
            self.color = (0, 255, 0)
            utils.handle_button_click(self.text)