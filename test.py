import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
# 进一步调高置信度，确保 i5 处理器只处理高质量的检测结果
hands = mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.9, 
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

def get_precision_count(hand_lms):
    fingers = []
    
    # --- 1. 大拇指判定（改用距离向量法） ---
    # 计算大拇指尖(4)到小指根部(17)的距离，张开手时这个距离最远
    dist_thumb_pinky = math.hypot(hand_lms.landmark[4].x - hand_lms.landmark[17].x, 
                                  hand_lms.landmark[4].y - hand_lms.landmark[17].y)
    # 计算大拇指根部(2)到小指根部(17)的参考距离
    dist_base = math.hypot(hand_lms.landmark[2].x - hand_lms.landmark[17].x, 
                           hand_lms.landmark[2].y - hand_lms.landmark[17].y)
    
    # 如果指尖距离明显大于指根距离，说明大拇指张开了
    if dist_thumb_pinky > dist_base * 1.3:
        fingers.append(1)
    else:
        fingers.append(0)

    # --- 2. 四根手指（改用中节指骨判定） ---
    # 判定逻辑：指尖(Tip)必须高于第二关节(PIP)且高于第三关节(MCP)
    tips = [8, 12, 16, 20]
    for tip in tips:
        # 增加高度差阈值 0.03，过滤掉握拳时的“假凸起”
        if hand_lms.landmark[tip].y < hand_lms.landmark[tip-2].y - 0.03:
            fingers.append(1)
        else:
            fingers.append(0)
            
    return fingers

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 针对笔记本摄像头优化曝光
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            f_list = get_precision_count(hand_lms)
            count = f_list.count(1)
            
            # 状态文字逻辑
            if count == 0:
                cv2.putText(frame, "STATUS: FIST", (50, 100), 2, 1.5, (0, 0, 255), 3)
            else:
                cv2.putText(frame, f"COUNT: {count}", (50, 100), 2, 1.5, (0, 255, 0), 3)

    cv2.imshow('Final Precision Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()