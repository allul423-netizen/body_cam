import cv2
import time
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. 初始化与环境配置 ---
load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL")
)

# 创建存储图片和日志的目录
if not os.path.exists('captures'): 
    os.makedirs('captures')
log_file = "activity_log.txt"

# 摄像头参数设置
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 重要：修复 NameError，在循环外定义初始变量 ---
ai_feedback = "System Initializing..." #
SAMPLE_INTERVAL = 5.0  # 考虑到 72B 模型响应较慢，设定为 5 秒采样一次
last_sample_time = time.time()

def draw_multiline_text(img, text, position, font_scale=0.6, color=(0, 255, 0), thickness=2):
    """在 OpenCV 窗口中绘制多行文本"""
    x, y0 = position
    line_height = 25 
    # 将文本按换行符拆分，逐行绘制
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * line_height
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def save_activity(frame, analysis):
    """保存图片并记录日志，实现文件名相互索引"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    img_filename = f"img_{timestamp}.jpg"
    img_path = os.path.join('captures', img_filename)
    
    # 保存当前帧图片
    cv2.imwrite(img_path, frame)
    
    # 写入日志文件
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] FILE: {img_filename} | ANALYSIS: {analysis}\n")
    return img_filename

print("AI 肢体动作监控已启动。按 'q' 退出...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) # 镜像翻转，更符合直觉
    current_time = time.time()

    # --- 2. 大模型采样与分析逻辑 ---
    if current_time - last_sample_time >= SAMPLE_INTERVAL:
        last_sample_time = current_time
        
        # 压缩图片质量，平衡上传速度与识别精度
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        base64_img = base64.b64encode(buffer).decode('utf-8')
        
        # 针对你的需求定制的高级 Prompt
        prompt = """Focus ONLY on human changes. Ignore other objects.
        1. Count extended fingers precisely.
        2. Check if eyes are open or closed.
        3. Describe body gesture details.
        Reply in concise English, strictly within 2 short lines."""

        try:
            # 使用 Qwen2-VL 视觉大模型
            response = client.chat.completions.create(
                model="Qwen/Qwen2-VL-72B-Instruct",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}]
            )
            raw_result = response.choices[0].message.content.strip()
            
            # 格式化输出：将句号替换为换行符以便显示
            ai_feedback = raw_result.replace(". ", ".\n")
            
            # 存储图片与日志索引
            save_activity(frame, raw_result)
            
        except Exception as e:
            ai_feedback = f"AI Analysis Error"

    # --- 3. UI 渲染与反馈 ---
    # 绘制半透明顶栏遮罩，增强文字可读性
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # 绘制多行 AI 反馈文本
    draw_multiline_text(frame, ai_feedback, (10, 30))
    
    cv2.imshow('AI Vision Body Monitor', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()