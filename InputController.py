from pynput.keyboard import Controller
from Event import EventBus, ButtonEvent, GestureEvent
import config
import utils


class InputController(Controller):
    def __init__(self):
        super().__init__()
        EventBus.subscribe(ButtonEvent.PRESS_CLICK, self.on_click)

    # 虚拟输入
    def on_click(self, data):
        button = data["button"]
        if config.input_mode != "chinese" and not button.is_func:
            self.press(button.text)

    # 处理按钮点击事件
    def button_function(self, button_text):
        """处理按钮点击事件"""
        if config.input_mode == "chinese":
            if button_text == "中/英":
                config.input_mode = "english"
                config.pinyin_input = ""
                config.candidates = []
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
            elif button_text == "删除":
                if config.pinyin_input:
                    config.pinyin_input = config.pinyin_input[:-1]
                    config.candidates = utils.get_pinyin_candidates(config.pinyin_input)
                    config.selected_index = 0
                else:
                    config.finalText = config.finalText[:-1] if config.finalText else ""
            elif button_text == "<":
                if config.pinyin_input:
                    config.selected_index = max(0, config.selected_index-1)
            elif button_text == ">":
                if config.pinyin_input:
                    config.selected_index = min(len(config.candidates)-1, config.selected_index+1)

            elif len(button_text) == 1 and button_text.isalpha():
                config.pinyin_input += button_text.lower()
                config.candidates = utils.get_pinyin_candidates(config.pinyin_input)
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