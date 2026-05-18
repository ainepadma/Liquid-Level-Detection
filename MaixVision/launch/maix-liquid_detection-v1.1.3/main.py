from maix import camera, display, image, nn, app, time, uart
from collections import deque

# ========== UART（只输出）==========
uart_dev = uart.UART("/dev/ttyS0", 115200)
uart_dev.write(b"MaixCAM YOLO ready\n")

# 定时检查用：存储锁定框颜色的简单对象
class _ObjChecker:
    _obj_lock_color = None
obj_checker = _ObjChecker()

# ========== 摄像头 ==========
CAM_W, CAM_H = 2560, 1440
cam = camera.Camera(CAM_W, CAM_H)

# ========== 显示 ==========
disp = display.Display()
DISP_W, DISP_H = disp.width(), disp.height()

# ========== 双模型 ==========
# 模型1: 自制模型 model_274449.mud
detector_a = nn.YOLOv5(model="/root/models/model_274449.mud", dual_buff=True)
print(f"Model A labels: {detector_a.labels}")

# 模型2: 通用 YOLOv5s
detector_b = nn.YOLOv5(model="/root/models/yolov5s.mud", dual_buff=True)
print(f"Model B labels: {detector_b.labels}")

# ========== 检测参数 ==========
# A 模型(自定义): 接受所有类别
TARGET_IDS_A = list(range(len(detector_a.labels)))
# B 模型(COCO): 只取 bottle(39), cup(41)
TARGET_IDS_B = [39, 41]
conf_th = 0.30
iou_th = 0.45

MIN_CUP_WIDTH = 30
MAX_CUP_WIDTH = CAM_W * 0.9
MIN_CUP_HEIGHT = 30
MAX_CUP_HEIGHT = CAM_H * 0.9

# ========== 输出降频 & 稳定性 ==========
OUTPUT_EVERY_N = 3               # 每 3 帧输出一次（加快响应）
OBJ_CHECK_INTERVAL = 15          # 定时检查
_frame_count = 0
_h_buf = deque(maxlen=OUTPUT_EVERY_N)
_t_buf = deque(maxlen=OUTPUT_EVERY_N)
_b_buf = deque(maxlen=OUTPUT_EVERY_N)
_conf_buf = deque(maxlen=OUTPUT_EVERY_N)
_color_buf = deque(maxlen=OUTPUT_EVERY_N)
_level_buf = deque(maxlen=OUTPUT_EVERY_N * 2)   # 液面距离缓冲（更大，稳定态时用）
_level_buf = deque(maxlen=OUTPUT_EVERY_N * 2)   # 液面距离缓冲（更大，稳定态时用）

# 稳定态 & 锁定
_prev_pos = None            # 上一帧检测框位置 (x, y, w, h)
_prev_level_y = -1          # 上一帧液面 Y（用于自适应基准色）
_level_locked_y = -1        # 锁定后的液面 Y
_level_lock_count = 0       # 液面锁定连续帧计数
_obj_locked = None          # 锁定后的物体检测框 (x, y, w, h)
_obj_lock_count = 0         # 物体锁定连续帧计数
_level_lock_window = 30     # 液面锁定后搜索范围
_obj_lock_window = 40       # 物体锁定后搜索范围（像素）
_stable_counter = 0         # 连续静止帧数（稳定态）
STABLE_THRESH = 15          # ~1秒无变化视为稳定
_obj_check_timer = 0        # 定时检查计数器

def uart_send(msg):
    """输出到 UART"""
    try:
        uart_dev.write((msg + "\n").encode())
    except:
        pass

while not app.need_exit():
    img = cam.read()

    # ===== 双模型推理（全屏）=====
    raw_a = detector_a.detect(img, conf_th=conf_th, iou_th=iou_th)
    raw_b = detector_b.detect(img, conf_th=conf_th * 0.6, iou_th=iou_th)

    # 合并两路结果
    candidates = []
    for obj in raw_a:
        if obj.class_id in TARGET_IDS_A:
            candidates.append((obj.score * 0.7, obj))
    for obj in raw_b:
        if obj.class_id in TARGET_IDS_B:
            candidates.append((obj.score, obj))

    # --- 筛选候选框 ---
    candidates_filtered = []
    for score, obj in candidates:
        if obj.class_id not in TARGET_IDS_A and obj.class_id not in TARGET_IDS_B:
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
        final_score = score * 5 + center_score * 3
        candidates_filtered.append((final_score, obj))

    # --- 2. 选最优候选 ---
    if candidates_filtered:
        candidates_filtered.sort(key=lambda x: x[0], reverse=True)
        best_obj = candidates_filtered[0][1]

        x, y, w, h = best_obj.x, best_obj.y, best_obj.w, best_obj.h
        conf = best_obj.score

        # 置信度标签（边缘精修前先用原始值画标签）
        label = f"{conf*100:.0f}%"
        tw, th = len(label) * 24, 40
        img.draw_rect(x, y-32, tw+6, th+6, image.COLOR_GREEN, thickness=-1)
        img.draw_rect(x, y-32, tw+6, th+6, image.COLOR_GREEN, thickness=3)
        img.draw_string(x+3, y-28, label, image.COLOR_BLACK, scale=3.0)

        # ===== 边缘精修：顶部 & 底部 =====
        mid_x = x + w // 2
        # 顶部精修：在 y-20 ~ y+20 范围内找最大亮度梯度
        best_top = y
        max_diff = 0
        for row in range(max(0, y-20), min(CAM_H, y+20), 2):
            v0 = img.get_pixel(mid_x, row)
            v1 = img.get_pixel(mid_x, row + 2)
            if isinstance(v0, (list, tuple)) and len(v0) >= 1: v0 = v0[0]
            if isinstance(v1, (list, tuple)) and len(v1) >= 1: v1 = v1[0]
            if isinstance(v0, int) and isinstance(v1, int):
                g0 = ((v0>>16)&0xFF) + ((v0>>8)&0xFF) + (v0&0xFF)
                g1 = ((v1>>16)&0xFF) + ((v1>>8)&0xFF) + (v1&0xFF)
                d = abs(g0 - g1)
                if d > max_diff: max_diff = d; best_top = row
        if max_diff > 20:
            y = best_top

        # 底部精修：在 y+h-20 ~ y+h+20 范围内找最大亮度梯度
        bot_base = y + h
        best_bot = bot_base
        max_diff = 0
        for row in range(max(0, bot_base-20), min(CAM_H, bot_base+20), 2):
            v0 = img.get_pixel(mid_x, row)
            v1 = img.get_pixel(mid_x, row + 2)
            if isinstance(v0, (list, tuple)) and len(v0) >= 1: v0 = v0[0]
            if isinstance(v1, (list, tuple)) and len(v1) >= 1: v1 = v1[0]
            if isinstance(v0, int) and isinstance(v1, int):
                g0 = ((v0>>16)&0xFF) + ((v0>>8)&0xFF) + (v0&0xFF)
                g1 = ((v1>>16)&0xFF) + ((v1>>8)&0xFF) + (v1&0xFF)
                d = abs(g0 - g1)
                if d > max_diff: max_diff = d; best_bot = row
        if max_diff > 20:
            h = best_bot - y

        # ===== 物体检测框锁定 =====
        _outlier_tolerance = 3        # 容忍连续 3 帧超限才解锁

        # 锁定状态下检查锁定框外是否有更高置信度的物体
        if _obj_locked is not None:
            lx, ly, lw, lh = _obj_locked
            # 当前锁定框内物体的置信度
            inside_score = conf

            # 找锁定框外的最高置信度候选
            outside_best = None
            outside_score = 0
            for sc_, obj_ in candidates_filtered:
                ox, oy, ow, oh = obj_.x, obj_.y, obj_.w, obj_.h
                if (ox > lx + lw or ox + ow < lx or oy > ly + lh or oy + oh < ly):
                    if obj_.score > outside_score:
                        outside_score = obj_.score
                        outside_best = obj_

            # 外部置信度明显高于框内 → 切换（外部比框内高 0.1 即可）
            if outside_best is not None and outside_score > inside_score + 0.1:
                x, y, w, h = outside_best.x, outside_best.y, outside_best.w, outside_best.h
                _obj_locked = None
                _level_locked_y = -1
                _obj_lock_count = 0
                _level_lock_count = 0
                _prev_pos = None

        if _obj_locked is not None:
            lx, ly, lw, lh = _obj_locked
            dx = abs(x - lx); dy = abs(y - ly); dw = abs(w - lw); dh = abs(h - lh)
            if dx < _obj_lock_window and dy < _obj_lock_window and dw < _obj_lock_window and dh < _obj_lock_window:
                x, y, w, h = lx, ly, lw, lh
                _obj_lock_count += 1
            else:
                # 锁定框与当前检测框不重合 → 检查是否有任何候选框与锁定框重合
                any_overlap = False
                for sc_, obj_ in candidates_filtered:
                    ox, oy, ow, oh = obj_.x, obj_.y, obj_.w, obj_.h
                    # 有重叠
                    if not (ox > lx + lw or ox + ow < lx or oy > ly + lh or oy + oh < ly):
                        any_overlap = True
                        break
                if not any_overlap:
                    # 没有任何检测框与锁定框重叠 → 物体已消失，立即解锁
                    _obj_locked = None
                    _level_locked_y = -1
                    _obj_lock_count = 0
                    _level_lock_count = 0
                else:
                    x, y, w, h = lx, ly, lw, lh
        else:
            if _prev_pos is not None:
                px, py, pw, ph = _prev_pos
                if abs(x-px)<_obj_lock_window and abs(y-py)<_obj_lock_window and abs(w-pw)<_obj_lock_window and abs(h-ph)<_obj_lock_window:
                    _obj_lock_count += 1
                else:
                    _obj_lock_count = 0
            if _obj_lock_count >= 10:
                _obj_locked = (x, y, w, h)

        # 重画检测框（锁定后的值）
        img.draw_rect(x, y, w, h, image.COLOR_GREEN, thickness=4)

        # ===== 定时物体存在性检查 =====
        _obj_check_timer += 1
        if _obj_locked is not None and _obj_check_timer >= OBJ_CHECK_INTERVAL:
            _obj_check_timer = 0
            # 在锁定框中心和四角采样，和之前记录的锁定框颜色比较
            old_r = old_g = old_b = -1
            if obj_checker._obj_lock_color is not None:
                old_r, old_g, old_b = obj_checker._obj_lock_color

            cr = cg = cb = cnt = 0
            lx, ly, lw, lh = _obj_locked
            for sx, sy in [(lx+lw//2, ly+lh//2), (lx+5, ly+5), (lx+lw-5, ly+5),
                           (lx+5, ly+lh-5), (lx+lw-5, ly+lh-5)]:
                if sx < 0 or sy < 0 or sx >= CAM_W or sy >= CAM_H: continue
                val = img.get_pixel(sx, sy)
                if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
                elif isinstance(val, int): v = val
                else: v = 0
                cr += (v>>16)&0xFF; cg += (v>>8)&0xFF; cb += v&0xFF; cnt += 1

            if cnt > 0:
                cr //= cnt; cg //= cnt; cb //= cnt
                if old_r >= 0:
                    d = ((cr-old_r)**2 + (cg-old_g)**2 + (cb-old_b)**2) ** 0.5
                    if d > 80:
                        _obj_locked = None
                        _level_locked_y = -1
                        _obj_lock_count = 0
                        _level_lock_count = 0
                obj_checker._obj_lock_color = (cr, cg, cb)

            # 额外解锁条件：锁定状态下连续多帧检测不到液面 → 锁定框可能错了
            if level_y < 0:
                _obj_lock_count = max(0, _obj_lock_count - 1)
                if _obj_lock_count <= 0:
                    _obj_locked = None
                    _level_locked_y = -1
                    _level_lock_count = 0
                # 锁定框中无可采样像素 → 物体消失
                _obj_locked = None
                _level_locked_y = -1
                _obj_lock_count = 0
                _level_lock_count = 0

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

        # ===== 自适应基准颜色 =====
        # 有历史液面时，取液面上方 1/3 处；否则用瓶底上方 h/10
        base_x = x + w // 2
        if _prev_level_y > 0:
            base_y = y + (_prev_level_y - y) // 3  # 液面上方 1/3（液体中上部）
        else:
            base_y = y + h - h // 10                # 瓶底上方 h/10（初始）
        base_r = base_g = base_b = 0
        base_m = w // 4
        for sx, sy in [(base_x, base_y), (base_x - base_m, base_y),
                       (base_x + base_m, base_y), (base_x, base_y + 5)]:
            val = img.get_pixel(sx, sy)
            if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
            elif isinstance(val, int): v = val
            else: v = 0
            base_r += (v >> 16) & 0xFF
            base_g += (v >> 8) & 0xFF
            base_b += v & 0xFF
        base_r //= 4; base_g //= 4; base_b //= 4

        # ===== 液面检测（色块粗定位 → 梯度扫描 → 置信度最高者胜出）=====
        level_y = -1
        roi_ly = y + h // 4
        roi_lh = h * 3 // 4
        if roi_lh > 30:
            liquid_th = [(0, 55, -30, 30, -30, 30)]
            blobs = img.find_blobs(liquid_th,
                                   roi=(x, roi_ly, w, roi_lh),
                                   x_stride=6, y_stride=3,
                                   area_threshold=500,
                                   pixels_threshold=250,
                                   merge=True)

            if blobs:
                best_blob = max(blobs, key=lambda b: b.area())
                rough_y = best_blob.y()

                scan_top = max(roi_ly, rough_y - 20)
                scan_bot = min(roi_ly + roi_lh, rough_y + 20)
                mid_x = x + w // 2
                area_cx = x + w // 2
                area_w = min(w // 3, 400)

                best_row = rough_y
                best_score = 0

                for row in range(scan_top, scan_bot, 2):
                    # 1) 梯度分
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

                    # 2) 下方颜色匹配分（靠近基准色 = 高分）
                    below_score = 0
                    for offset in (12, 24):
                        cr = cg = cb = cnt = 0
                        cy = row + offset
                        if cy + 6 <= y + h:
                            for sr in range(cy, cy + 6, 4):
                                for sc in range(area_cx - area_w//2, area_cx + area_w//2, 16):
                                    val = img.get_pixel(sc, sr)
                                    if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
                                    elif isinstance(val, int): v = val
                                    else: continue
                                    cr += (v>>16)&0xFF; cg += (v>>8)&0xFF; cb += v&0xFF; cnt += 1
                            if cnt > 2:
                                cr //= cnt; cg //= cnt; cb //= cnt
                                d = ((cr-base_r)**2+(cg-base_g)**2+(cb-base_b)**2)**0.5
                                below_score += max(0, 1.0 - d/150)   # d=0→1分, d=150→0分

                    # 3) 上方颜色分（远离基准色 = 高分，即浅色）
                    above_score = 0
                    cr = cg = cb = cnt = 0
                    for sr in range(row - 25, row - 8, 4):
                        for sc in range(area_cx - area_w//2, area_cx + area_w//2, 16):
                            val = img.get_pixel(sc, sr)
                            if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
                            elif isinstance(val, int): v = val
                            else: continue
                            cr += (v>>16)&0xFF; cg += (v>>8)&0xFF; cb += v&0xFF; cnt += 1
                    if cnt > 3:
                        cr //= cnt; cg //= cnt; cb //= cnt
                        d = ((cr-base_r)**2+(cg-base_g)**2+(cb-base_b)**2)**0.5
                        above_score = min(1.0, d / 150)        # d=150→1分, d=0→0分

                    score = diff_sum * 0.5 + below_score * 80 + above_score * 50
                    if score > best_score:
                        best_score = score
                        best_row = row

                if best_score > 20:
                    level_y = best_row

        # ===== 液面锁定 =====
        if level_y > 0:
            if _level_locked_y > 0:
                if abs(level_y - _level_locked_y) < _level_lock_window:
                    # 锁定范围内：直接保持锁定值（防漂移）
                    level_y = _level_locked_y
                    _level_lock_count += 1
                else:
                    # 超出范围：扣分，连续超限则解锁
                    _level_lock_count = max(0, _level_lock_count - 2)
                    if _level_lock_count <= 0:
                        _level_locked_y = -1
                    else:
                        level_y = _level_locked_y
            else:
                if _prev_level_y > 0 and abs(level_y - _prev_level_y) < 15:
                    _level_lock_count += 1
                else:
                    _level_lock_count = 0
                if _level_lock_count >= 10:
                    _level_locked_y = level_y

            _prev_level_y = level_y

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

    time.sleep_ms(10)