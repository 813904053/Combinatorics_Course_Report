from Event import EventBus, GestureEvent
import config
import utils

class GestureHandler:
    def __init__(self):
        self.gesture_cooldown = 0
        self.setup_event_handlers()
        self.confidence = 0

        # 持续时间相关
        self.threshold_frames = 20
        self.last_gesture = None
        self.counter = 0
        self.allow_recognition = False

    def setup_event_handlers(self):
        """注册手势事件处理器"""
        # 握拳
        EventBus.subscribe(GestureEvent.FIST, self.on_fist)
        # 手掌张开
        EventBus.subscribe(GestureEvent.OPEN_HAND, self.on_open_hand)
        # 点赞 - 确认
        EventBus.subscribe(GestureEvent.THUMBS_UP, self.on_thumbs_up)
        # 左大拇指
        EventBus.subscribe(GestureEvent.THUMBS_LEFT, self.on_thumbs_left)
        # 掌
        EventBus.subscribe(GestureEvent.PALM, self.on_palm)
        # SPIDER
        EventBus.subscribe(GestureEvent.SPIDER, self.on_spider)

    def can_trigger_gesture(self, gesture_type):
        """检查是否可以触发手势（防重复）"""
        if self.gesture_cooldown > 0:
            return False
        if gesture_type == self.last_gesture:
            return False

        self.last_gesture = gesture_type
        self.gesture_cooldown = 20  # 冷却帧数
        return True

    def update(self, current_gesture):
        """每帧更新"""
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1

    def check_gesture_duration(self, current_gesture):
        """检查手势持续时间"""
        if current_gesture == self.last_gesture:
            self.counter += 1
            #print(self.counter)
        else:
            self.last_gesture = current_gesture
            self.counter = 1

        # 如果持续帧数达到阈值
        if self.counter >= self.threshold_frames:
            # 重置计数器避免重复触发
            return True
        return False

    def on_fist(self, data):
        if self.can_trigger_gesture("fist"):
            print(f"手势: fist")

    def on_open_hand(self, data):
        if self.last_gesture == "1":
            print(f"手势: open_hand")
            self.counter = 0

            # 以下是1帧触发，现改成多帧触发
            #config.can_type_in = True
            #self.allow_recognition = False

    def on_thumbs_up(self, data):
        if self.last_gesture == "1":
            # 确认逻辑
            print("手势: thumbs_up")
            if config.candidates:
                config.finalText += config.candidates[config.selected_index]
                config.pinyin_input = ""
                config.candidates = []
                config.selected_index = 0



    def on_thumbs_left(self, current_gesture):
        if self.last_gesture == "1":
            print("手势: thumbs_left")
            if config.pinyin_input:
                config.pinyin_input = config.pinyin_input[:-1]
                config.candidates = utils.get_pinyin_candidates(config.pinyin_input)
            else:
                config.finalText = config.finalText[:-1] if config.finalText else ""

    def on_palm(self, current_gesture):
        if self.last_gesture == "1":
            print("手势: on_palm")
            if config.pinyin_input:
                config.pinyin_input = ""
                config.candidates = []
            else:
                config.finalText = ""

    def on_spider(self, current_gesture):
        if self.last_gesture == "1":
            print("手势: on_spider")
            if config.input_mode == "chinese":
                config.input_mode = "english"
            else:
                config.input_mode = "chinese"