
# 事件类型
class ButtonEvent:
    IDLE_HOVER = "hover_start"
    HOVER_IDLE = "hover_end"
    HOVER_PRESS = "press_start"
    PRESS_IDLE = "press_end"
    PRESS_CLICK = "click_start"
    CLICK_IDLE = "click_end"

# 在 Event.py 中添加
class GestureEvent:
    THUMBS_UP = "thumbs_up"
    FIST = "fist"
    OPEN_HAND = "open_hand"
    THUMBS_LEFT = "thumbs_left"
    PALM = "palm"
    SPIDER = "spider"


# 事件总线（发布-订阅）
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