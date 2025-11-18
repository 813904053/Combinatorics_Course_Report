from pynput.keyboard import Controller
from Event import EventBus, ButtonEvent, GestureEvent
import config

class InputController(Controller):
    def __init__(self):
        super().__init__()
        EventBus.subscribe(ButtonEvent.PRESS_CLICK, self.on_click)

    def on_click(self, data):
        button = data["button"]
        if config.input_mode != "chinese" and not button.is_func:
            self.press(button.text)
