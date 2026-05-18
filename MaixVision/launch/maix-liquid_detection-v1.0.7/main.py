from maix import camera, display, image, nn, app, time, uart
from collections import deque

# ========== UART（只输出）==========
uart_dev = uart.UART("/dev/ttyS0", 115200)
uart_dev.write(b"MaixCAM YOLO ready\n")

# ========== 摄像头 ==========
CAM_W, CAM_H = 1440, 1440
cam = camera.Camera(CAM_W, CAM_H)

# ========== 显示 ==========
disp = display.Display()
DISP_W, DISP_H = disp.width(), disp.height()

# ========== 模型 ==========
detector = nn.YOLOv5(model="/root/models/yolov5s.mud", dual_buff=True)

# ========== 检测参数 ==========
TARGET_IDS = [39, 41]           # bottle(39), cup(41)
conf_th = 0.30
iou_th = 0.45

MIN_CUP_WIDTH = 30
MAX_CUP_WIDTH = CAM_W * 0.9
MIN_CUP_HEIGHT = 30
MAX_CUP_HEIGHT = CAM_H * 0.9

# ========== 输出降频 ==========
OUTPUT_EVERY_N = 5               # 每 5 帧输出一次平均值
_frame_count = 0
_h_buf = deque(maxlen=OUTPUT_EVERY_N)
_t_buf = deque(maxlen=OUTPUT_EVERY_N)
_b_buf = deque(maxlen=OUTPUT_EVERY_N)
_conf_buf = deque(maxlen=OUTPUT_EVERY_N)
_color_buf = deque(maxlen=OUTPUT_EVERY_N)
_class_buf = deque(maxlen=OUTPUT_EVERY_N)

def uart_send(msg):
    """输出到 UART"""
    try:
        uart_dev.write((msg + "\n").encode())
    except:
        pass

while not app.need_exit():
    img = cam.read()
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
        score = obj.score * 5 + center_score * 3

        candidates.append((score, obj))

    # --- 2. 选最优候选 ---
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_obj = candidates[0][1]

        x, y, w, h = best_obj.x, best_obj.y, best_obj.w, best_obj.h
        conf = best_obj.score

        # 颜色检测
        cx, cy = x + w // 2, y + h // 2
        inset_w, inset_h = w // 4, h // 4
        r_sum = g_sum = b_sum = 0
        for sx, sy in [(cx,cy), (x+inset_w, y+inset_h), (x+w-inset_w, y+inset_h),
                        (x+inset_w, y+h-inset_h), (x+w-inset_w, y+h-inset_h)]:
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

        img.draw_rect(x, y, w, h, image.COLOR_GREEN, thickness=4)

        # 在识别框上标注名称和置信度（绿色框+白字）
        cls_name = "bottle" if best_obj.class_id == 39 else "cup"
        label = f"{cls_name} {conf*100:.0f}%"

        # 绿色背景框 + 绿色边框 + 白色文字
        tw, th = len(label) * 30, 45
        img.draw_rect(x, y-36, tw+8, th+8, image.COLOR_GREEN, thickness=-1)
        img.draw_rect(x, y-36, tw+8, th+8, image.COLOR_GREEN, thickness=3)
        img.draw_string(x+4, y-32, label, image.COLOR_BLACK, scale=3.5)

        # 累积数据到缓冲区
        _h_buf.append(h)
        _t_buf.append(y)
        _b_buf.append(CAM_H - (y + h))
        _conf_buf.append(conf)
        _color_buf.append(hex_color)
        _class_buf.append(best_obj.class_id)
        _frame_count += 1

        # 每 N 帧输出一次平均值
        if _frame_count >= OUTPUT_EVERY_N and len(_h_buf) == OUTPUT_EVERY_N:
            avg_h = sum(_h_buf) // len(_h_buf)
            avg_t = sum(_t_buf) // len(_t_buf)
            avg_b = sum(_b_buf) // len(_b_buf)
            avg_c = sum(_conf_buf) / len(_conf_buf)
            avg_cls = max(set(_class_buf), key=_class_buf.count)
            # 颜色取 R、G、B 分别平均
            rr = gg = bb = 0
            for c in _color_buf:
                rr += int(c[1:3], 16)
                gg += int(c[3:5], 16)
                bb += int(c[5:7], 16)
            avg_color = f"#{rr//5:02X}{gg//5:02X}{bb//5:02X}"

            # 输出格式: class,h,T,B,conf,color
            # class: 39=bottle 41=cup    h: 像素高度    T: 上边距(px)    B: 下边距(px)
            # conf: 置信度(0~1)    color: 中心区域平均颜色(RRGGBB)
            uart_send(f"{avg_cls},{avg_h},{avg_t},{avg_b},{avg_c:.2f},{avg_color}")
            _frame_count = 0

    # ===== 显示 =====
    img_disp = img.resize(DISP_W, DISP_H)
    disp.show(img_disp)

    time.sleep_ms(60)