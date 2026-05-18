from maix import camera, display, image, nn, app, time, uart
from collections import deque

# ========== UART（只输出）==========
uart_dev = uart.UART("/dev/ttyS0", 115200)
uart_dev.write(b"MaixCAM YOLO ready\n")

# ========== 摄像头 ==========
CAM_W, CAM_H = 2560, 1440
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

# ========== 输出降频 & 稳定性 ==========
OUTPUT_EVERY_N = 5
_frame_count = 0
_h_buf = deque(maxlen=OUTPUT_EVERY_N)
_t_buf = deque(maxlen=OUTPUT_EVERY_N)
_b_buf = deque(maxlen=OUTPUT_EVERY_N)
_conf_buf = deque(maxlen=OUTPUT_EVERY_N)
_color_buf = deque(maxlen=OUTPUT_EVERY_N)
_level_buf = deque(maxlen=OUTPUT_EVERY_N * 2)   # 液面距离缓冲（更大，稳定态时用）

# 稳定态检测
_prev_pos = None            # 上一帧检测框位置 (x, y, w, h)
_stable_counter = 0         # 连续静止帧数
STABLE_THRESH = 15          # ~1秒(60ms*15≈900ms)无变化视为稳定

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

        img.draw_rect(x, y, w, h, image.COLOR_GREEN, thickness=4)

        # 置信度标签
        label = f"{conf*100:.0f}%"
        tw, th = len(label) * 24, 40
        img.draw_rect(x, y-32, tw+6, th+6, image.COLOR_GREEN, thickness=-1)
        img.draw_rect(x, y-32, tw+6, th+6, image.COLOR_GREEN, thickness=3)
        img.draw_string(x+3, y-28, label, image.COLOR_BLACK, scale=3.0)

        # ===== 稳定态检测 =====
        cur_pos = (x, y, w, h)
        if _prev_pos is not None:
            # 位置变化 < 2% 视为静止
            dx = abs(x - _prev_pos[0])
            dy = abs(y - _prev_pos[1])
            dw = abs(w - _prev_pos[2])
            dh = abs(h - _prev_pos[3])
            if dx < CAM_W * 0.02 and dy < CAM_H * 0.02 and dw < w * 0.02 and dh < h * 0.02:
                _stable_counter += 1
            else:
                _stable_counter = 0
        _prev_pos = cur_pos

        # ===== 液面检测（色块粗定位 + 像素扫描精修）=====
        level_y = -1
        roi_ly = y + h // 2
        roi_lh = h // 2
        if roi_lh > 30:
            liquid_th = [(0, 55, -30, 30, -30, 30)]
            blobs = img.find_blobs(liquid_th,
                                   roi=(x, roi_ly, w, roi_lh),
                                   x_stride=4, y_stride=2,
                                   area_threshold=300,
                                   pixels_threshold=150,
                                   merge=True)

            if blobs:
                best_blob = max(blobs, key=lambda b: b.area())
                # 粗定位：色块顶部
                rough_y = best_blob.y()

                # 精修：在 rough_y 上下 30px 范围内找最大亮度梯度
                scan_top = max(roi_ly, rough_y - 20)
                scan_bot = min(roi_ly + roi_lh, rough_y + 20)
                mid_x = x + w // 2
                max_diff = 0
                best_row = rough_y

                for row in range(scan_top, scan_bot, 2):
                    diff_sum = 0
                    for col in (mid_x - w//4, mid_x, mid_x + w//4):
                        v0 = img.get_pixel(col, row)
                        v1 = img.get_pixel(col, row + 2)
                        if isinstance(v0, (list, tuple)) and len(v0) >= 1: v0 = v0[0]
                        if isinstance(v1, (list, tuple)) and len(v1) >= 1: v1 = v1[0]
                        if isinstance(v0, int) and isinstance(v1, int):
                            g0 = ((v0>>16)&0xFF) + ((v0>>8)&0xFF) + (v0&0xFF)
                            g1 = ((v1>>16)&0xFF) + ((v1>>8)&0xFF) + (v1&0xFF)
                            diff_sum += abs(g0 - g1)
                    if diff_sum > max_diff:
                        max_diff = diff_sum
                        best_row = row

                if max_diff > 30:
                    level_y = best_row

        # 画液面线（红色）
        if level_y > 0:
            bottle_bottom = y + h
            img.draw_line(x, level_y, x + w, level_y, image.COLOR_RED, thickness=4)
            mid_x = x + w // 2
            img.draw_arrow(mid_x, level_y, mid_x, bottle_bottom, image.COLOR_RED, thickness=2)
            dist_level = bottle_bottom - level_y
            img.draw_string(mid_x + 5, (level_y + bottle_bottom)//2,
                          f"{dist_level}px", image.COLOR_RED, scale=2.0)

        # ===== 液体颜色检测（液面下方液体中心区域）=====
        if level_y > 0:
            liq_cx = x + w // 2
            liq_cy = level_y + (bottle_bottom - level_y) // 2  # 液面和瓶底中间
            liq_margin = min(w, bottle_bottom - level_y) // 4
            r_sum = g_sum = b_sum = 0
            pts = [(liq_cx, liq_cy),
                   (liq_cx - liq_margin, liq_cy),
                   (liq_cx + liq_margin, liq_cy),
                   (liq_cx, liq_cy - liq_margin//2),
                   (liq_cx, liq_cy + liq_margin//2)]
            for sx, sy in pts:
                val = img.get_pixel(sx, sy)
                if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
                elif isinstance(val, int): v = val
                else: v = 0
                r_sum += (v >> 16) & 0xFF
                g_sum += (v >> 8) & 0xFF
                b_sum += v & 0xFF
            hex_color = f"#{r_sum//5:02X}{g_sum//5:02X}{b_sum//5:02X}"
        else:
            hex_color = "#000000"

        # 累积数据到缓冲区
        _h_buf.append(h)
        _t_buf.append(y)
        _b_buf.append(CAM_H - (y + h))
        _conf_buf.append(conf)
        _color_buf.append(hex_color)
        if level_y > 0:
            _level_buf.append(bottle_bottom - level_y)
        _frame_count += 1

        # 每 N 帧输出一次
        if _frame_count >= OUTPUT_EVERY_N and len(_h_buf) == OUTPUT_EVERY_N:
            is_stable = _stable_counter >= STABLE_THRESH

            if is_stable:
                # 稳定态：用中值替代均值，剔除异常
                sh = sorted(_h_buf); avg_h = sh[len(sh)//2]
                st = sorted(_t_buf); avg_t = st[len(st)//2]
                sb = sorted(_b_buf); avg_b = sb[len(sb)//2]
                sc = sorted(_conf_buf); avg_c = sc[len(sc)//2]
            else:
                avg_h = sum(_h_buf) // len(_h_buf)
                avg_t = sum(_t_buf) // len(_t_buf)
                avg_b = sum(_b_buf) // len(_b_buf)
                avg_c = sum(_conf_buf) / len(_conf_buf)

            rr = gg = bb = 0
            for c in _color_buf:
                rr += int(c[1:3], 16)
                gg += int(c[3:5], 16)
                bb += int(c[5:7], 16)
            avg_color = f"#{rr//5:02X}{gg//5:02X}{bb//5:02X}"

            # 液面距离（稳定态用中值）
            if len(_level_buf) > 0:
                if is_stable and len(_level_buf) >= 5:
                    sl = sorted(_level_buf)
                    # 剔除两端的离群点
                    trim = max(1, len(sl) // 5)
                    core = sl[trim:-trim] if len(sl) > trim*2 else sl
                    avg_level = sum(core) // len(core)
                else:
                    avg_level = sum(_level_buf) // len(_level_buf)
                _level_buf.clear()
            else:
                avg_level = -1

            # 输出: h,T,B,conf,color,level_dist
            # h: 像素高度  T: 上边距(px)  B: 下边距(px)
            # conf: 置信度(0~1)  color: 中心区域平均颜色(RRGGBB)  level_dist: 液面到底部距离(px)
            uart_send(f"{avg_h},{avg_t},{avg_b},{avg_c:.2f},{avg_color},{avg_level}")
            _frame_count = 0

    # ===== 显示 =====
    img_disp = img.resize(DISP_W, DISP_H)
    disp.show(img_disp)

    time.sleep_ms(60)