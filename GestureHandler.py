from Event import EventBus, GestureEvent
import config
import utils

class GestureHandler:
    def __init__(self):
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.setup_event_handlers()

    def setup_event_handlers(self):
        """注册手势事件处理器"""
        # 点赞 - 确认
        EventBus.subscribe(GestureEvent.THUMBS_UP, self.on_thumbs_up)
        # 握拳 - 删除
        EventBus.subscribe(GestureEvent.FIST, self.on_fist)
        # 手掌张开 - 空格
        EventBus.subscribe(GestureEvent.OPEN_HAND, self.on_open_hand)
        # 比耶 - 清空
        EventBus.subscribe(GestureEvent.VICTORY, self.on_victory)
        # OK - 切换输入法
        EventBus.subscribe(GestureEvent.OK, self.on_ok)

    def can_trigger_gesture(self, gesture_type):
        """检查是否可以触发手势（防重复）"""
        if self.gesture_cooldown > 0:
            return False
        if gesture_type == self.last_gesture:
            return False

        self.last_gesture = gesture_type
        self.gesture_cooldown = 20  # 冷却帧数
        return True

    def update(self):
        """每帧更新冷却"""
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1

    def on_thumbs_up(self, data):
        if self.can_trigger_gesture("thumbs_up"):
            # 确认逻辑
            if config.candidates:
                config.finalText += config.candidates[config.selected_index]
                config.pinyin_input = ""
                config.candidates = []
                config.selected_index = 0
            print("事件: 手势确认")

    def on_fist(self, data):
        if self.can_trigger_gesture("fist"):
            # 删除逻辑
            if config.pinyin_input:
                config.pinyin_input = config.pinyin_input[:-1]
                config.candidates = utils.get_pinyin_candidates(config.pinyin_input)
            else:
                config.finalText = config.finalText[:-1] if config.finalText else ""
            print("事件: 手势删除")

    def on_open_hand(self, data):
        if self.can_trigger_gesture("open_hand"):
            # 空格逻辑
            config.finalText += " "
            print("事件: 手势空格")

    def on_victory(self, data):
        if self.can_trigger_gesture("victory"):
            # 清空逻辑
            config.finalText = ""
            config.pinyin_input = ""
            config.candidates = []
            config.selected_index = 0
            print("事件: 手势清空")

    def on_ok(self, data):
        if self.can_trigger_gesture("ok"):
            # 切换输入法
            config.input_mode = "english" if config.input_mode == "chinese" else "chinese"
            config.pinyin_input = ""
            config.candidates = []
            print("事件: 手势切换输入法")