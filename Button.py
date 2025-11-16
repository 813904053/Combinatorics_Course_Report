import cv2
import numpy as np

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text
        self.state = "normal"
        self.tilted_corners = None
        self.tilted_center = None

        self.points = np.float32([
            [self.pos[0], self.pos[1]],
            [self.pos[0] + self.size[0], self.pos[1]],
            [self.pos[0] + self.size[0], self.pos[1] + self.size[1]],
            [self.pos[0], self.pos[1] + self.size[1]]
        ])

    def pos_adjust(self, keyboard_pos, offset):
        self.pos = [self.pos[0] + keyboard_pos[0] + offset,
                      self.pos[1] + keyboard_pos[1] + offset]
        self.points = np.float32([
            [self.pos[0], self.pos[1]],
            [self.pos[0] + self.size[0], self.pos[1]],
            [self.pos[0] + self.size[0], self.pos[1] + self.size[1]],
            [self.pos[0], self.pos[1] + self.size[1]]
        ])

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