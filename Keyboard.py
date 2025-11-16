import config
import numpy as np


class Keyboard:
    def __init__(self, buttonList):
        self.w = 1100
        self.h = 450
        self.buttonList = buttonList
        self.pos = [(config.width - self.w) // 2, config.height - self.h - 50]

        # 键盘的四个点
        self.points = np.float32([
            [self.pos[0], self.pos[1]],
            [self.pos[0] + self.w, self.pos[1]],
            [self.pos[0] + self.w, self.pos[1] + self.h],
            [self.pos[0], self.pos[1] + self.h]
        ])

        self.button_offset = 50
        # 按键坐标校正，让按键坐标转换为绝对坐标
        for button in buttonList:
            button.pos_adjust(self.pos, self.button_offset)