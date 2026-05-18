from maix import camera, display, image, nn, app, time
import sys, select
from collections import deque

# ========== 摄像头（先初始化，避免模型抢占内存导致 VI pool 失败）==========
CAM_W, CAM_H = 1440, 1440          # 1:1 方形采集，与显示缩放一致，减少 resize 开销
cam = camera.Camera(CAM_W, CAM_H)

# ========== 显示 ==========
disp = display.Display()
DISP_W, DISP_H = disp.width(), disp.height()
print(f"显示分辨率: {DISP_W}x{DISP_H}")

# ========== 模型（后加载，摄像头已占好 VI pool）==========
model_path = "/root/models/yolov5s.mud"
detector = nn.YOLOv5(model=model_path, dual_buff=True)
MODEL_INPUT_W = detector.input_width()
MODEL_INPUT_H = detector.input_height()
print(f"模型输入: {MODEL_INPUT_W}x{MODEL_INPUT_H}")

# ========== 检测参数 ==========
TARGET_IDS = [39, 41]           # bottle(39), cup(41)
conf_th = 0.30
iou_th = 0.45

# 尺寸过滤阈值——适配 1440x1440 方形
MIN_CUP_WIDTH = 30               # 放宽最小宽度，适配矮小杯子
MAX_CUP_WIDTH = CAM_W * 0.9
MIN_CUP_HEIGHT = 30              # 放宽最小高度，原来 80 太高过滤掉了矮杯子
MAX_CUP_HEIGHT = CAM_H * 0.9

# ========== 平滑窗口 ==========
WINDOW_SIZE = 5
width_window = deque(maxlen=WINDOW_SIZE)
height_window = deque(maxlen=WINDOW_SIZE)

# 记录上一帧的稳定尺寸，用于当前帧的目标筛选
prev_stable_w = 0
prev_stable_h = 0

def read_cmd():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

print("类别标签:", detector.labels)
print("命令: conf <值> / get / q")

while not app.need_exit():
    img = cam.read()

    # ===== YOLO 检测：横屏 2560x1440 直接检测 =====
    detect_res = detector.detect(img, conf_th=conf_th, iou_th=iou_th)

    # --- 1. 筛选杯子候选框 ---
    candidates = []
    for obj in detect_res:
        if obj.class_id not in TARGET_IDS:
            continue

        x, y, w, h = obj.x, obj.y, obj.w, obj.h
        if w < MIN_CUP_WIDTH or w > MAX_CUP_WIDTH:
            continue
        if h < MIN_CUP_HEIGHT or h > MAX_CUP_HEIGHT:
            continue

        if prev_stable_w > 0 and prev_stable_h > 0:
            size_diff = abs(w - prev_stable_w) / prev_stable_w + abs(h - prev_stable_h) / prev_stable_h
            score = obj.score * 10 - size_diff
        else:
            score = obj.score * 10

        candidates.append((score, obj))

    # --- 2. 选最优候选 ---
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_obj = candidates[0][1]

        x, y, w, h = best_obj.x, best_obj.y, best_obj.w, best_obj.h
        conf = best_obj.score

        # 计算物体边框到图像边界的距离
        dist_left   = x                    # 左边距
        dist_right  = CAM_W - (x + w)      # 右边距
        dist_top    = y                    # 上边距
        dist_bottom = CAM_H - (y + h)      # 下边距

        print(f"[detect] class={best_obj.class_id}  w={w}  h={h}  conf={conf:.3f}  "
              f"L={dist_left} R={dist_right} T={dist_top} B={dist_bottom}")

        width_window.append(w)
        height_window.append(h)

        # 显示屏只画侦测框
        img.draw_rect(x, y, w, h, image.COLOR_GREEN, thickness=4)

        # --- 3. 平滑尺寸（终端打印，不画在屏幕上）---
        sorted_w = sorted(width_window)
        sorted_h = sorted(height_window)
        mid = len(sorted_w) // 2
        stable_w = sorted_w[mid]
        stable_h = sorted_h[mid]

        prev_stable_w = stable_w
        prev_stable_h = stable_h
    else:
        pass

    # ===== 显示 =====
    img_disp = img.resize(DISP_W, DISP_H)
    disp.show(img_disp)

    # --- 4. 串口命令 ---
    cmd = read_cmd()
    if cmd:
        parts = cmd.strip().split()
        if parts[0] == "conf" and len(parts) == 2:
            try:
                new_conf = float(parts[1])
                if 0.0 <= new_conf <= 1.0:
                    conf_th = new_conf
                    print(f"置信度阈值: {conf_th}")
            except:
                pass
        elif cmd == "get":
            if len(width_window) > 0:
                print(f"稳定中值 -> 宽: {prev_stable_w:.1f} px, 高: {prev_stable_h:.1f} px")
            else:
                print("当前未检测到目标")
        elif cmd == "q":
            app.set_exit_flag(True)

    time.sleep_ms(60)