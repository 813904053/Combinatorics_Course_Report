import time
import cv2

# 显示部分
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# 帧率
fps_start_time = time.time()
frame_count = 0
while True:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    # 每100帧计算一次帧率
    frame_count += 1
    if frame_count % 100 == 0:
        fps = frame_count / (time.time() - fps_start_time)
        frame_count = 0
        fps_start_time = time.time()
    # 在图像上显示FPS（每帧都显示）
    current_fps = frame_count / (time.time() - fps_start_time) if frame_count > 0 else 0
    cv2.putText(img, f"FPS: {current_fps:.1f}", (1000, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Chinese Virtual Keyboard", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break