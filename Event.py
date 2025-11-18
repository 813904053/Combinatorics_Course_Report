
# 事件类型
class ButtonEvent:
    IDLE_HOVER = "hover_start"
    HOVER_IDLE = "hover_end"
    HOVER_PRESS = "press_start"
    PRESS_IDLE = "press_end"
    PRESS_CLICK = "click"

# 在 Event.py 中添加
class GestureEvent:
    THUMBS_UP = "thumbs_up"        # 点赞
    FIST = "fist"                  # 握拳
    OPEN_HAND = "open_hand"        # 手掌张开
    VICTORY = "victory"            # 比耶
    OK = "ok"                      # OK手势



# 事件总线（简单的发布-订阅）
class EventBus:
    listeners = {}

    @classmethod
    def subscribe(cls, event_type, callback):
        if event_type not in cls.listeners:
            cls.listeners[event_type] = []
        cls.listeners[event_type].append(callback)

    @classmethod
    def publish(cls, event_type, data):
        for callback in cls.listeners.get(event_type, []):
            callback(data)