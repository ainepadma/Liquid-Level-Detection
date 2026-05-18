from maix import camera, display, image, nn, app, time, uart
from collections import deque
import os

# ========== UART 串口（TX/RX 收发命令，波特率 115200）==========
uart_dev = uart.UART("/dev/ttyS0", 115200)

# ========== 摄像头（先初始化，避免模型抢占内存导致 VI pool 失败）==========
CAM_W, CAM_H = 1440, 1440          # 1:1 方形采集
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

MIN_CUP_WIDTH = 30
MAX_CUP_WIDTH = CAM_W * 0.9
MIN_CUP_HEIGHT = 30
MAX_CUP_HEIGHT = CAM_H * 0.9

# ========== 平滑窗口 ==========
WINDOW_SIZE = 5
width_window = deque(maxlen=WINDOW_SIZE)
height_window = deque(maxlen=WINDOW_SIZE)
prev_stable_w = 0
prev_stable_h = 0

# ========== 标定 ==========
# 用法：把已知高度的物体放在固定距离处，输入 b <真实高度_mm> 如 b 200
calib_px_per_mm = 0.0
calib_ready = False

# ========== UART 命令收发 ==========
uart_dev = uart.UART("/dev/ttyS0", 115200)

def read_cmd():
    """非阻塞读取命令（UART + 文件管道）"""
    # 方式1: UART（先检查是否有数据）
    try:
        if uart_dev.any():              # 有数据才读，避免阻塞
            line = uart_dev.readline()
            if line:
                return line.strip()
    except:
        pass
    # 方式2: 文件管道（shell: echo b > /tmp/maix_cmd）
    try:
        if os.path.exists("/tmp/maix_cmd"):
            with open("/tmp/maix_cmd", "r") as f:
                cmd = f.read().strip()
            os.remove("/tmp/maix_cmd")
            return cmd if cmd else None
    except:
        pass
    return None

def uart_print(msg):
    """输出到终端 + UART"""
    print(msg)
    try:
        uart_dev.write(msg + "\n")
    except:
        pass

print("类别标签:", detector.labels)
print("命令: b<高度mm>标定 / conf <值> / get / q=退出  例: b200")

while not app.need_exit():
    img = cam.read()

    # ===== YOLO 检测 =====
    detect_res = detector.detect(img, conf_th=conf_th, iou_th=iou_th)

    # --- 1. 筛选候选框 ---
    candidates = []
    for obj in detect_res:
        if obj.class_id not in TARGET_IDS:
            continue
        x, y, w, h = obj.x, obj.y, obj.w, obj.h
        if w < MIN_CUP_WIDTH or w > MAX_CUP_WIDTH:
            continue
        if h < MIN_CUP_HEIGHT or h > MAX_CUP_HEIGHT:
            continue

        obj_cx = x + w // 2
        obj_cy = y + h // 2
        center_dist = ((obj_cx - CAM_W//2)**2 + (obj_cy - CAM_H//2)**2) ** 0.5
        max_dist = ((CAM_W//2)**2 + (CAM_H//2)**2) ** 0.5
        center_score = 1.0 - (center_dist / max_dist)

        if prev_stable_w > 0 and prev_stable_h > 0:
            size_diff = abs(w - prev_stable_w) / prev_stable_w + abs(h - prev_stable_h) / prev_stable_h
            score = obj.score * 5 + center_score * 3 - size_diff
        else:
            score = obj.score * 5 + center_score * 3

        candidates.append((score, obj))

    # --- 2. 选最优候选 ---
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_obj = candidates[0][1]

        x, y, w, h = best_obj.x, best_obj.y, best_obj.w, best_obj.h
        conf = best_obj.score

        dist_top    = y
        dist_bottom = CAM_H - (y + h)

        # 颜色检测
        cx, cy = x + w // 2, y + h // 2
        inset_w, inset_h = w // 4, h // 4
        sample_pts = [
            (cx, cy),
            (x + inset_w, y + inset_h),
            (x + w - inset_w, y + inset_h),
            (x + inset_w, y + h - inset_h),
            (x + w - inset_w, y + h - inset_h),
        ]
        r_sum = g_sum = b_sum = 0
        for sx, sy in sample_pts:
            val = img.get_pixel(sx, sy)
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                v = val[0]
            elif isinstance(val, int):
                v = val
            else:
                v = 0
            r_sum += (v >> 16) & 0xFF
            g_sum += (v >> 8) & 0xFF
            b_sum += v & 0xFF
        hex_color = f"#{r_sum//5:02X}{g_sum//5:02X}{b_sum//5:02X}"

        width_window.append(w)
        height_window.append(h)

        # 平滑
        sorted_w = sorted(width_window)
        sorted_h = sorted(height_window)
        mid = len(sorted_w) // 2
        stable_w = sorted_w[mid]
        stable_h = sorted_h[mid]
        prev_stable_w = stable_w
        prev_stable_h = stable_h

        # === 检测显示 ===
        img.draw_rect(x, y, w, h, image.COLOR_GREEN, thickness=4)
        if calib_ready:
            real_h = h / calib_px_per_mm if calib_px_per_mm > 0 else 0
            uart_print(f"[detect] class={best_obj.class_id}  h_px={h}  real_h={real_h:.0f}mm  "
                  f"conf={conf:.3f}  T={dist_top}  B={dist_bottom}  color={hex_color}")
        else:
            uart_print(f"[detect] class={best_obj.class_id}  h_px={h}  conf={conf:.3f}  "
                  f"T={dist_top}  B={dist_bottom}  color={hex_color}  (未标定)")
    else:
        pass

    # ===== 显示 =====
    img_disp = img.resize(DISP_W, DISP_H)
    disp.show(img_disp)

    # ===== 串口命令 =====
    cmd = read_cmd()
    if cmd:
        parts = cmd.strip().split()
        if parts[0].startswith("b") and len(parts[0]) > 1:
            # 标定: b<真实高度_mm>  例: b200
            try:
                real_h_mm = float(parts[0][1:])
                if real_h_mm > 0 and candidates:
                    _, best = max(candidates, key=lambda x: x[0])
                    h_now = best.h
                    calib_px_per_mm = h_now / real_h_mm
                    calib_ready = True
                    uart_print(f">>> 标定完成!  h_px={h_now}  @{real_h_mm:.0f}mm  比例={calib_px_per_mm:.3f} px/mm")
                else:
                    print(">>> 标定失败: 未检测到物体")
            except:
                print(">>> 格式: b<高度mm>  例: b200")

        elif parts[0] == "conf" and len(parts) == 2:
            try:
                new_conf = float(parts[1])
                if 0.0 <= new_conf <= 1.0:
                    conf_th = new_conf
                    print(f"置信度阈值: {conf_th}")
            except:
                pass

        elif cmd == "get":
            if len(width_window) > 0:
                if calib_ready:
                    real_h = stable_h / calib_px_per_mm
                    print(f"稳定中值 -> 像素: {stable_w:.0f}x{stable_h:.0f}  真实高度: {real_h:.0f}mm")
                else:
                    print(f"稳定中值 -> 像素: {stable_w:.0f}x{stable_h:.0f}  (未标定)")
            else:
                print("当前未检测到目标")

        elif cmd == "q":
            app.set_exit_flag(True)

    time.sleep_ms(60)