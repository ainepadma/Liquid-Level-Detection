from maix import camera, display, image, nn, app, time, uart
from collections import deque

# ========== UART（只输出）==========
try:
    uart_dev = uart.UART("/dev/ttyS0", 115200)
    uart_dev.write(b"MaixCAM YOLO ready\n")
except Exception as e:
    print(f"UART FAIL: {e}")
    uart_dev = None

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
_level_lock_window = 40     # 液面锁定后搜索范围
_obj_lock_window = 60       # 物体锁定后搜索范围（像素）
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
    # 如果所有候选置信度都 < 0.2，拒绝此次检测
    if candidates_filtered and all(obj.score < 0.2 for _, obj in candidates_filtered):
        candidates_filtered = []
    has_obj = bool(candidates_filtered)
    level_y = -1
    if has_obj:
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

        # 边缘精修（梯度+颜色权重）
        mid_x = x + w // 2
        # 顶部参考色
        top_ref_r=top_ref_g=top_ref_b=0
        for i in range(3):
            val=img.get_pixel(mid_x, max(0, y-15+i*3))
            if isinstance(val,(list,tuple)) and len(val)>=1: v=val[0]
            elif isinstance(val,int): v=val
            else: v=0
            top_ref_r+=(v>>16)&0xFF; top_ref_g+=(v>>8)&0xFF; top_ref_b+=v&0xFF
        top_ref_r//=3; top_ref_g//=3; top_ref_b//=3

        best_top=y; best_score=0
        for row in range(max(0,y-20), min(CAM_H,y+20), 2):
            v0=img.get_pixel(mid_x,row); v1=img.get_pixel(mid_x,row+2)
            if isinstance(v0,(list,tuple)) and len(v0)>=1: v0=v0[0]
            if isinstance(v1,(list,tuple)) and len(v1)>=1: v1=v1[0]
            if isinstance(v0,int) and isinstance(v1,int):
                g0=(v0>>16)&0xFF+(v0>>8)&0xFF+(v0&0xFF)
                g1=(v1>>16)&0xFF+(v1>>8)&0xFF+(v1&0xFF)
                grad=abs(g0-g1)
                cd=((((v1>>16)&0xFF)-top_ref_r)**2+(((v1>>8)&0xFF)-top_ref_g)**2+((v1&0xFF)-top_ref_b)**2)**0.5
                score=grad*0.9+max(0,1.0-cd/200)*10
                if score>best_score: best_score=score; best_top=row
        if best_score>12: y=best_top

        bot_base=y+h; best_bot=bot_base; best_score=0
        bot_ref_r=bot_ref_g=bot_ref_b=0
        for i in range(3):
            val=img.get_pixel(mid_x, max(0,bot_base-10+i*3))
            if isinstance(val,(list,tuple)) and len(val)>=1: v=val[0]
            elif isinstance(val,int): v=val
            else: v=0
            bot_ref_r+=(v>>16)&0xFF; bot_ref_g+=(v>>8)&0xFF; bot_ref_b+=v&0xFF
        bot_ref_r//=3; bot_ref_g//=3; bot_ref_b//=3
        for row in range(max(0,bot_base-20), min(CAM_H,bot_base+20), 2):
            v0=img.get_pixel(mid_x,row); v1=img.get_pixel(mid_x,row+2)
            if isinstance(v0,(list,tuple)) and len(v0)>=1: v0=v0[0]
            if isinstance(v1,(list,tuple)) and len(v1)>=1: v1=v1[0]
            if isinstance(v0,int) and isinstance(v1,int):
                g0=(v0>>16)&0xFF+(v0>>8)&0xFF+(v0&0xFF)
                g1=(v1>>16)&0xFF+(v1>>8)&0xFF+(v1&0xFF)
                grad=abs(g0-g1)
                # 用瓶底基准色（后面会算）- 先用一个粗估值
                cd=((((v1>>16)&0xFF)-bot_ref_r)**2+(((v1>>8)&0xFF)-bot_ref_g)**2+((v1&0xFF)-bot_ref_b)**2)**0.5
                score=grad*0.9+max(0,1.0-cd/200)*10
                if score>best_score: best_score=score; best_bot=row
        if best_score>12: h=best_bot-y

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
            # 未锁定：指数滑动平均逼近（快速稳定）
            if _prev_pos is not None:
                px, py, pw, ph = _prev_pos
                # 用 EMA 逐步逼近：新值 = 旧值*0.7 + 检测值*0.3
                x = int(px * 0.7 + x * 0.3)
                y = int(py * 0.7 + y * 0.3)
                w = int(pw * 0.7 + w * 0.3)
                h = int(ph * 0.7 + h * 0.3)
                # 如果逼近后的值与上次差异 < 窗口，计数
                if abs(x-px)<_obj_lock_window and abs(y-py)<_obj_lock_window and abs(w-pw)<_obj_lock_window and abs(h-ph)<_obj_lock_window:
                    _obj_lock_count += 1
                else:
                    _obj_lock_count = 0
            if _obj_lock_count >= 8:
                _obj_locked = (x, y, w, h)

        # 重画检测框（锁定后的值）
        img.draw_rect(x, y, w, h, image.COLOR_GREEN, thickness=8)

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
                _obj_lock_count = max(0, _obj_lock_count - 1)  # 每帧超限只扣 1
                if _obj_lock_count <= 0:
                    _obj_locked = None
                    _level_locked_y = -1
                    _level_lock_count = 0
                # 锁定框中无可采样像素 → 物体消失
                _obj_locked = None
                _level_locked_y = -1
                _obj_lock_count = 0
                _level_lock_count = 0

        # ===== 液面检测（改进版）=====
        base_x = x + w // 2
        # ---------- 自适应基准色（多行采样，中值去噪）----------
        if _prev_level_y > 0 and _prev_level_y < y + h:
            # 以前一帧液面下方区域为参考
            sample_start = _prev_level_y + 10
            sample_end = min(y + h, _prev_level_y + 50)
            col_offsets = [0, -w//4, w//4, -w//6, w//6]
            colors = []
            for sr in range(sample_start, sample_end, 6):
                for sc_off in col_offsets:
                    sx = base_x + sc_off
                    if 0 <= sx < CAM_W:
                        val = img.get_pixel(sx, sr)
                        if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
                        elif isinstance(val, int): v = val
                        else: continue
                        colors.append(((v>>16)&0xFF, (v>>8)&0xFF, v&0xFF))
            if colors:
                # 按亮度排序取中值，抗噪
                colors.sort(key=lambda c: sum(c))
                mid_idx = len(colors)//2
                base_r, base_g, base_b = colors[mid_idx]
            else:
                base_r = base_g = base_b = 0
        else:
            # 初始基准色：瓶底上方 10% 处多点采样（避开瓶底反光）
            base_y_start = y + h - h//7
            base_y_end = y + h - h//15
            col_offsets = [0, -w//4, w//4, -w//5, w//5]
            colors = []
            for sy in range(base_y_start, base_y_end, 5):
                for sc_off in col_offsets:
                    sx = base_x + sc_off
                    if 0 <= sx < CAM_W:
                        val = img.get_pixel(sx, sy)
                        if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
                        elif isinstance(val, int): v = val
                        else: continue
                        colors.append(((v>>16)&0xFF, (v>>8)&0xFF, v&0xFF))
            if colors:
                colors.sort(key=lambda c: sum(c))
                base_r, base_g, base_b = colors[len(colors)//2]
            else:
                base_r = base_g = base_b = 0

        roi_ly = y + h // 4
        roi_lh = h * 3 // 4
        level_y = -1
        if roi_lh > 30:
            # 扩大 blob 检测范围，降低漏检
            blobs = img.find_blobs([(0, 55, -30, 30, -30, 30)], 
                                   roi=(x, roi_ly, w, roi_lh),
                                   x_stride=5, y_stride=2, area_threshold=400, 
                                   pixels_threshold=200, merge=True)
            if blobs:
                # 取面积前 3 的 blob 中位 y 作为粗定位
                blobs_sorted = sorted(blobs, key=lambda b: b.area(), reverse=True)[:3]
                rough_y = sum(b.y() for b in blobs_sorted) // len(blobs_sorted)
                # 扩大扫描范围，上下各延展 35 像素
                scan_top = max(roi_ly, rough_y - 35)
                scan_bot = min(roi_ly + roi_lh, rough_y + 35)
            else:
                # 无 blob 时，扫描整个 ROI 的上半部（液面通常在上方）
                scan_top = roi_ly
                scan_bot = roi_ly + roi_lh // 2

            mid_x = x + w // 2
            area_cx = mid_x
            area_w = min(w//3, 400)
            candidates = []          # 存储 (score, row)

            for row in range(scan_top, scan_bot, 2):
                # 垂直梯度（三列）
                diff_sum = 0
                for col in (mid_x - w//4, mid_x, mid_x + w//4):
                    v0 = img.get_pixel(col, row)
                    v1 = img.get_pixel(col, row+2)
                    if isinstance(v0, (list, tuple)) and len(v0) >= 1: v0 = v0[0]
                    if isinstance(v1, (list, tuple)) and len(v1) >= 1: v1 = v1[0]
                    if isinstance(v0, int) and isinstance(v1, int):
                        diff_sum += abs(((v0>>16)&0xFF)+((v0>>8)&0xFF)+(v0&0xFF) -
                                        ((v1>>16)&0xFF)-((v1>>8)&0xFF)-(v1&0xFF))

                # 下方颜色采样（3 段，偏移 14, 30, 48）
                below_scores = []
                below_colors = []
                for offset in (14, 30, 48):
                    cr = cg = cb = cnt = 0
                    cy = row + offset
                    if cy + 4 <= y + h:
                        for sr in range(cy, cy+4, 2):
                            for sc in range(area_cx - area_w//2, area_cx + area_w//2, 12):
                                val = img.get_pixel(sc, sr)
                                if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
                                elif isinstance(val, int): v = val
                                else: continue
                                cr += (v>>16)&0xFF; cg += (v>>8)&0xFF; cb += v&0xFF; cnt += 1
                        if cnt > 1:
                            cr //= cnt; cg //= cnt; cb //= cnt
                            d = ((cr-base_r)**2 + (cg-base_g)**2 + (cb-base_b)**2) ** 0.5
                            below_scores.append(max(0, 1.0 - d/120))
                            below_colors.append((cr, cg, cb))
                        else:
                            below_scores.append(0)
                            below_colors.append((0,0,0))
                    else:
                        below_scores.append(0)
                        below_colors.append((0,0,0))

                # 下方颜色一致性
                consistency = 0
                if len(below_colors) >= 2:
                    diffs = []
                    for i in range(len(below_colors)):
                        for j in range(i+1, len(below_colors)):
                            if below_colors[i] != (0,0,0) and below_colors[j] != (0,0,0):
                                d = ((below_colors[i][0]-below_colors[j][0])**2 +
                                     (below_colors[i][1]-below_colors[j][1])**2 +
                                     (below_colors[i][2]-below_colors[j][2])**2) ** 0.5
                                diffs.append(d)
                    if diffs:
                        avg_diff = sum(diffs) / len(diffs)
                        consistency = max(0, 1.0 - avg_diff / 300)
                # 下方总体颜色匹配得分
                below_score = sum(below_scores) / len(below_scores) if below_scores else 0

                # 渐变方向：越往下越接近基准色
                gradient_bonus = 0
                if len(below_colors) >= 2 and below_colors[0] != (0,0,0) and below_colors[-1] != (0,0,0):
                    d1 = ((below_colors[0][0]-base_r)**2+(below_colors[0][1]-base_g)**2+(below_colors[0][2]-base_b)**2)**0.5
                    d_last = ((below_colors[-1][0]-base_r)**2+(below_colors[-1][1]-base_g)**2+(below_colors[-1][2]-base_b)**2)**0.5
                    if d1 > 0:
                        gradient_bonus = max(0, (d1 - d_last) / d1) * 2.0

                # 连续性：左右梯度差异
                side_diff = 0
                for col in (x + w//4, x + w*3//4):
                    v0 = img.get_pixel(col, row)
                    v1 = img.get_pixel(col, row+2)
                    if isinstance(v0, (list, tuple)) and len(v0) >= 1: v0 = v0[0]
                    if isinstance(v1, (list, tuple)) and len(v1) >= 1: v1 = v1[0]
                    if isinstance(v0, int) and isinstance(v1, int):
                        side_diff += abs(((v0>>16)&0xFF)+((v0>>8)&0xFF)+(v0&0xFF) -
                                         ((v1>>16)&0xFF)-((v1>>8)&0xFF)-(v1&0xFF))
                continuity = max(0, 1.0 - abs(diff_sum - side_diff) / max(diff_sum, 1))

                # 上方颜色（用于对比度计算）
                above_r = above_g = above_b = cnt2 = 0
                for sr in range(row-18, row-4, 4):
                    for sc in range(area_cx - area_w//2, area_cx + area_w//2, 12):
                        val = img.get_pixel(sc, sr)
                        if isinstance(val, (list, tuple)) and len(val) >= 1: v = val[0]
                        elif isinstance(val, int): v = val
                        else: continue
                        above_r += (v>>16)&0xFF; above_g += (v>>8)&0xFF; above_b += v&0xFF; cnt2 += 1
                if cnt2 > 3: above_r //= cnt2; above_g //= cnt2; above_b //= cnt2
                else: above_r = above_g = above_b = 0

                contrast = 0.5
                if below_colors and below_colors[0] != (0,0,0) and above_r+above_g+above_b > 0:
                    br, bg, bb = below_colors[0]
                    diff_ab = abs(br-above_r) + abs(bg-above_g) + abs(bb-above_b)
                    denom = max(br+bg+bb, above_r+above_g+above_b, 1)
                    contrast = min(1.0, diff_ab / denom)

                # 动态权重
                w_grad = 0.3 + (1.0 - contrast) * 1.0
                w_col  = 0.8 + contrast * 3.0
                w_con  = 0.4 + contrast * 0.8

                # 严重偏离基准色时减分（避免选到瓶壁）
                color_penalty = 1.0 if below_score > 0.25 else 0.3

                score = (below_score * 180 * w_col +
                         gradient_bonus * 150 +
                         diff_sum * 50 * w_grad +
                         consistency * 60 * w_con +
                         continuity * 10) * color_penalty

                if score > 5:   # 最小阈值过滤
                    candidates.append((score, row))

            if candidates:
                # 选得分最高的行
                candidates.sort(key=lambda t: t[0], reverse=True)
                best_row = candidates[0][1]
                best_score = candidates[0][0]

                # 若存在略低但得分相近且对比度更高的行，向下微调
                for sc, row in candidates[1:]:
                    if row > best_row and sc >= best_score * 0.75:
                        # 用简单的对比度检查
                        above_val = 0; below_val = 0
                        if best_row > 0 and best_row + 2 < CAM_H:
                            v_a = img.get_pixel(mid_x, best_row-2)
                            v_b = img.get_pixel(mid_x, best_row+2)
                            if isinstance(v_a, (list,tuple)) and len(v_a)>=1: v_a=v_a[0]
                            if isinstance(v_b, (list,tuple)) and len(v_b)>=1: v_b=v_b[0]
                            if isinstance(v_a,int) and isinstance(v_b,int):
                                above_val = ((v_a>>16)&0xFF)+((v_a>>8)&0xFF)+(v_a&0xFF)
                                below_val = ((v_b>>16)&0xFF)+((v_b>>8)&0xFF)+(v_b&0xFF)
                        if abs(above_val - below_val) > 30:
                            best_row = row
                            best_score = sc

                # 液面最终微调
                level_y = min(best_row, y + h - 2)

        # ===== 稳定态检测（仅 has_obj 内有效）=====
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

        # ===== 液面锁定（仅物体锁定后才锁定液面）=====
        if level_y > 0 and _obj_locked is not None:
            if _level_locked_y > 0:
                if abs(level_y - _level_locked_y) < _level_lock_window:
                    # 锁定范围内：直接保持锁定值（防漂移）
                    level_y = _level_locked_y
                    _level_lock_count += 1
                else:
                    # 超出范围：扣分，连续超限则解锁
                    _level_lock_count = max(0, _level_lock_count - 1)  # 每帧超限只扣 1
                    if _level_lock_count <= 0:
                        _level_locked_y = -1
                    else:
                        level_y = _level_locked_y
            else:
                if _prev_level_y > 0 and abs(level_y - _prev_level_y) < 15:
                    _level_lock_count += 1
                else:
                    _level_lock_count = 0
                if _level_lock_count >= 8:
                    _level_locked_y = level_y

            _prev_level_y = level_y
        elif _obj_locked is None:
            # 物体未锁定：重置液面锁定状态
            _level_locked_y = -1
            _level_lock_count = 0

        # 画液面线（红色加粗 + 箭头）
        if level_y > 0:
            bottle_bottom = y + h
            img.draw_line(x, level_y, x + w, level_y, image.COLOR_RED, thickness=8)
            mid_x = x + w // 2
            img.draw_arrow(mid_x, level_y, mid_x, bottle_bottom, image.COLOR_RED, thickness=5)
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

    # ===== 屏幕右上角状态栏（统一在循环末尾绘制）=====
    if _obj_locked is not None:
        obj_status = "LOCKED"
        obj_color = image.COLOR_GREEN
    elif candidates_filtered:
        obj_status = "UNLOCK"
        obj_color = image.COLOR_YELLOW
    else:
        obj_status = "NONE"
        obj_color = image.COLOR_RED

    # level_y 在主循环中定义，非物体帧不可用，默认 NONE
    try:
        _ = level_y
    except:
        level_y = -1
    if level_y > 0:
        if _level_locked_y > 0:
            liq_status = "LOCKED"
            liq_color = image.COLOR_GREEN
        else:
            liq_status = "UNLOCK"
            liq_color = image.COLOR_YELLOW
    else:
        liq_status = "NONE"
        liq_color = image.COLOR_RED

    sx, sy = CAM_W - 800, 10
    bw, bh = 680, 104
    img.draw_rect(sx, sy, bw, bh, obj_color, thickness=-1)
    img.draw_rect(sx, sy, bw, bh, obj_color, thickness=6)
    img.draw_string(sx + 16, sy + 16, f"Obj:{obj_status}", image.COLOR_BLACK, scale=7.0)
    sy2 = sy + bh + 16
    img.draw_rect(sx, sy2, bw, bh, liq_color, thickness=-1)
    img.draw_rect(sx, sy2, bw, bh, liq_color, thickness=6)
    img.draw_string(sx + 16, sy2 + 16, f"Liq:{liq_status}", image.COLOR_BLACK, scale=7.0)

    # ===== 显示 =====
    img_disp = img.resize(DISP_W, DISP_H)
    disp.show(img_disp)

    time.sleep_ms(10)