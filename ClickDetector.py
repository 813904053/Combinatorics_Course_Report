from Event import EventBus, ButtonEvent

class ClickDetector:
    def __init__(self, id):
        self.id = id
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
                EventBus.publish(ButtonEvent.IDLE_HOVER, {
                    "button": hovered_button,
                    "finger_id": self.id
                })

        elif self.state == "HOVER":
            if hovered_button is None or hovered_button != self.current_button:
                self.state = "IDLE"
                EventBus.publish(ButtonEvent.HOVER_IDLE, {
                    "button": self.current_button,
                    "finger_id": self.id
                })
                self.current_button = None

            else:
                if self.prev_finger_y is not None:
                    move_distance = current_finger_y - self.prev_finger_y
                    if move_distance > 2:
                        self.total_press_distance += move_distance
                    if self.total_press_distance > self.press_threshold:
                        EventBus.publish(ButtonEvent.HOVER_PRESS, {         # HOVER-->PRESS
                            "button": self.current_button,
                            "finger_id": self.id
                        })
                        self.state = "PRESS"


                self.prev_finger_y = current_finger_y

        elif self.state == "PRESS":
            if hovered_button is None or hovered_button != self.current_button:
                EventBus.publish(ButtonEvent.PRESS_IDLE, {
                    "button": self.current_button,
                    "finger_id": self.id
                })
                self.state = "IDLE"
                self.total_press_distance = 0
                self.current_button = None

            else:
                if self.prev_finger_y is not None:
                    move_distance = current_finger_y - self.prev_finger_y
                    if move_distance < -5:
                        EventBus.publish(ButtonEvent.PRESS_CLICK, {
                            "button": self.current_button,
                            "finger_id": self.id
                        })
                        self.state = "CLICK"
                    elif move_distance > 0:
                        self.total_press_distance += move_distance
                self.prev_finger_y = current_finger_y

        elif self.state == "CLICK":
            self.state = "IDLE"
            self.current_button = None
            self.prev_finger_y = None
            self.total_press_distance = 0

        self.prev_finger_y = current_finger_y
        return None