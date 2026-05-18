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

# ========== 检测/锁定参数（瓶子/杯子） ==========
# 检测阈值
DETECT_CONF_TH = 0.30           # 模型A基础阈值
DETECT_IOU_TH = 0.45
DETECT_B_CONF_SCALE = 0.60      # 模型B阈值倍率

# 杯子/瓶子分开阈值
CUP_CONF_MIN = 0.20
BOTTLE_CONF_MIN = 0.30

# 目标框尺寸范围
MIN_OBJ_WIDTH = 30
MAX_OBJ_WIDTH = CAM_W * 0.9
MIN_OBJ_HEIGHT = 30
MAX_OBJ_HEIGHT = CAM_H * 0.9

# 候选打分权重
CAND_SCORE_W = 5.0
CAND_CENTER_W = 2.0
CAND_PREV_W = 4.0
CAND_LEVEL_W = 3.0            # 色块/液面接近度权重（靠近加分）
LEVEL_PROX_SCALE = 200.0      # 接近度距离缩放（像素），越大越不敏感
LEVEL_EMA_ALPHA = 0.45        # 色块接受后用于迭代更新液面位置的 EMA 权重

# 锁定逻辑参数
LOCK_IOU_MIN = 0.30
LOCK_FAST_CONF = 0.45
LOCK_FRAMES_FAST = 3
LOCK_FRAMES_NORMAL = 5
LOCK_MISS_MAX = 6
LOCK_EMA_ALPHA = 0.7
LOCK_EMA_BETA = 0.3

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
_base_r = 0
_base_g = 0
_base_b = 0
_base_valid = False
_obj_miss_count = 0

# 本地基准色默认值（液面检测前必定义）
base_r = base_g = base_b = 0

def uart_send(msg):
    """输出到 UART"""
    try:
        uart_dev.write((msg + "\n").encode())
    except:
        pass

def calc_iou(ax, ay, aw, ah, bx, by, bw, bh):
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

while not app.need_exit():
    img = cam.read()

    # ===== 双模型推理（全屏）=====
    raw_a = detector_a.detect(img, conf_th=DETECT_CONF_TH, iou_th=DETECT_IOU_TH)
    raw_b = detector_b.detect(img, conf_th=DETECT_CONF_TH * DETECT_B_CONF_SCALE, iou_th=DETECT_IOU_TH)

    # 合并两路结果
    candidates = []
    for obj in raw_a:
        if obj.class_id in TARGET_IDS_A:
            label = detector_a.labels[obj.class_id] if obj.class_id < len(detector_a.labels) else ""
            candidates.append((obj.score * 0.7, obj, label))
    for obj in raw_b:
        if obj.class_id in TARGET_IDS_B:
            label = detector_b.labels[obj.class_id] if obj.class_id < len(detector_b.labels) else ""
            candidates.append((obj.score, obj, label))

    # --- 筛选候选框 ---
    candidates_filtered = []
    prev_cx = prev_cy = None
    if _prev_pos is not None:
        prev_cx = _prev_pos[0] + _prev_pos[2] // 2
        prev_cy = _prev_pos[1] + _prev_pos[3] // 2
    for score, obj, label in candidates:
        if obj.class_id not in TARGET_IDS_A and obj.class_id not in TARGET_IDS_B:
            continue
        min_conf = CUP_CONF_MIN if label == "cup" else BOTTLE_CONF_MIN
        if obj.score < min_conf:
            continue
        x, y, w, h = obj.x, obj.y, obj.w, obj.h
        if w < MIN_OBJ_WIDTH or w > MAX_OBJ_WIDTH:
            continue
        if h < MIN_OBJ_HEIGHT or h > MAX_OBJ_HEIGHT:
            continue

        obj_cx = x + w // 2
        obj_cy = y + h // 2
        center_dist = ((obj_cx - CAM_W//2)**2 + (obj_cy - CAM_H//2)**2) ** 0.5
        max_dist = ((CAM_W//2)**2 + (CAM_H//2)**2) ** 0.5
        center_score = 1.0 - (center_dist / max_dist)
        prev_score = 0.0
        if prev_cx is not None and prev_cy is not None:
            prev_dist = ((obj_cx - prev_cx) ** 2 + (obj_cy - prev_cy) ** 2) ** 0.5
            prev_score = 1.0 - (prev_dist / max_dist)
            if prev_score < 0:
                prev_score = 0.0
        # 色块/液面接近度：使用上一帧液面或锁定液面作为参考（靠近加分）
        level_est = _level_locked_y if _level_locked_y > 0 else (_prev_level_y if _prev_level_y > 0 else -1)
        level_prox = 0.0
        if level_est > 0:
            dist_level = abs(y - level_est)
            level_prox = max(0.0, 1.0 - (dist_level / LEVEL_PROX_SCALE))

        final_score = score * CAND_SCORE_W + center_score * CAND_CENTER_W + prev_score * CAND_PREV_W + CAND_LEVEL_W * level_prox
        candidates_filtered.append((final_score, obj))

    # --- 2. 选最优候选 ---
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
        if _obj_locked is not None:
            lx, ly, lw, lh = _obj_locked
            best_iou = 0.0
            best_obj = None
            for _, obj_ in candidates_filtered:
                ox, oy, ow, oh = obj_.x, obj_.y, obj_.w, obj_.h
                iou = calc_iou(lx, ly, lw, lh, ox, oy, ow, oh)
                if iou > best_iou:
                    best_iou = iou
                    best_obj = obj_

            if best_obj is not None and best_iou >= 0.25:
                nx, ny, nw, nh = best_obj.x, best_obj.y, best_obj.w, best_obj.h
                x = int(lx * 0.7 + nx * 0.3)
                y = int(ly * 0.7 + ny * 0.3)
                w = int(lw * 0.7 + nw * 0.3)
                h = int(lh * 0.7 + nh * 0.3)
                _obj_locked = (x, y, w, h)
                _obj_miss_count = 0
            else:
                x, y, w, h = lx, ly, lw, lh
                _obj_miss_count += 1
                if _obj_miss_count >= 6:
                    _obj_locked = None
                    _obj_miss_count = 0
                    _level_locked_y = -1
                    _level_lock_count = 0
                    _prev_pos = None
        else:
            # 未锁定：指数滑动平均逼近（快速稳定）
            if _prev_pos is not None:
                px, py, pw, ph = _prev_pos
                x = int(px * 0.7 + x * 0.3)
                y = int(py * 0.7 + y * 0.3)
                w = int(pw * 0.7 + w * 0.3)
                h = int(ph * 0.7 + h * 0.3)
                iou = calc_iou(px, py, pw, ph, x, y, w, h)
                if iou >= 0.30:
                    _obj_lock_count += 1
                else:
                    _obj_lock_count = max(0, _obj_lock_count - 1)
            else:
                _obj_lock_count = 1

            lock_need = 3 if conf >= 0.45 else 5
            if _obj_lock_count >= lock_need:
                _obj_locked = (x, y, w, h)
                _obj_miss_count = 0

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

        # ===== 基准色：由液面与物体底部之间的区域确定（在液面检测前，用上一帧液面估计值） =====
        if _prev_level_y > 0 and _obj_locked is not None:
            blk_bottom = y + h
            region_top = max(y, _prev_level_y)
            region_h = blk_bottom - region_top
            if region_h >= 20:
                sr_sum = sg_sum = sb_sum = scnt = 0
                mid_x = x + w // 2
                sample_w = w // 3
                for yy in range(region_top, blk_bottom, 8):
                    for xx in range(max(x, mid_x - sample_w//2), min(x + w, mid_x + sample_w//2), 12):
                        if xx < 0 or xx >= CAM_W or yy < 0 or yy >= CAM_H:
                            continue
                        v = img.get_pixel(xx, yy)
                        if isinstance(v, (list, tuple)) and len(v) >= 1: v = v[0]
                        elif not isinstance(v, int): continue
                        sr_sum += (v >> 16) & 0xFF
                        sg_sum += (v >> 8) & 0xFF
                        sb_sum += v & 0xFF
                        scnt += 1
                if scnt > 0:
                    new_r = sr_sum // scnt
                    new_g = sg_sum // scnt
                    new_b = sb_sum // scnt
                    if not _base_valid:
                        _base_r, _base_g, _base_b = new_r, new_g, new_b
                        _base_valid = True
                    else:
                        _base_r = int(_base_r * 0.85 + new_r * 0.15)
                        _base_g = int(_base_g * 0.85 + new_g * 0.15)
                        _base_b = int(_base_b * 0.85 + new_b * 0.15)

        if _base_valid:
            base_r, base_g, base_b = _base_r, _base_g, _base_b

        # ===== 液面检测 =====
        roi_ly = y + h // 4; roi_lh = h * 3 // 4
        if roi_lh > 30:
            blobs = img.find_blobs([(0, 55, -30, 30, -30, 30)], roi=(x, roi_ly, w, roi_lh),
                                   x_stride=6, y_stride=3, area_threshold=500, pixels_threshold=250, merge=True)
            if blobs:
                best_blob = max(blobs, key=lambda b: b.area())
                rough_y = best_blob.y()
                scan_top = max(roi_ly, rough_y-20); scan_bot = min(roi_ly+roi_lh, rough_y+20)
                mid_x = x + w // 2; area_cx = x + w // 2; area_w = min(w//3, 400)
                best_row = rough_y; best_score = 0; best_contrast = 0
                second_row = -1; second_score = -1; second_contrast = 0
                for row in range(scan_top, scan_bot, 2):
                    diff_sum = 0
                    for col in (mid_x-w//4, mid_x, mid_x+w//4):
                        v0 = img.get_pixel(col, row); v1 = img.get_pixel(col, row+2)
                        if isinstance(v0, (list, tuple)) and len(v0) >= 1: v0 = v0[0]
                        if isinstance(v1, (list, tuple)) and len(v1) >= 1: v1 = v1[0]
                        if isinstance(v0, int) and isinstance(v1, int):
                            diff_sum += abs(((v0>>16)&0xFF)+((v0>>8)&0xFF)+(v0&0xFF) - ((v1>>16)&0xFF)-((v1>>8)&0xFF)-(v1&0xFF))
                    below_score = 0
                    # 取下方3段颜色，同时计算一致性
                    below_colors = []
                    for offset in (12, 32, 52):
                        cr=cg=cb=cnt=0; cy=row+offset
                        if cy+6 <= y+h:
                            for sr in range(cy, cy+6, 4):
                                for sc in range(area_cx-area_w//2, area_cx+area_w//2, 16):
                                    val = img.get_pixel(sc, sr)
                                    if isinstance(val, (list, tuple)) and len(val) >= 1: v=val[0]
                                    elif isinstance(val, int): v=val
                                    else: continue
                                    cr+=(v>>16)&0xFF; cg+=(v>>8)&0xFF; cb+=v&0xFF; cnt+=1
                            if cnt>2:
                                cr//=cnt; cg//=cnt; cb//=cnt
                                d=((cr-base_r)**2+(cg-base_g)**2+(cb-base_b)**2)**0.5
                                below_score += max(0, 1.0-d/120)
                                below_colors.append((cr, cg, cb))

                    # 下方颜色一致性：3段之间的颜色变化越小越好
                    consistency = 0
                    if len(below_colors) >= 2:
                        for i in range(len(below_colors)):
                            for j in range(i+1, len(below_colors)):
                                c1, c2 = below_colors[i], below_colors[j]
                                d = ((c1[0]-c2[0])**2+(c1[1]-c2[1])**2+(c1[2]-c2[2])**2)**0.5
                                consistency += d
                        consistency = max(0, 1.0 - consistency / 300)  # 越一致越高分

                    side_diff = 0
                    for col in (x+w//4, x+w*3//4):
                        v0=img.get_pixel(col,row); v1=img.get_pixel(col,row+2)
                        if isinstance(v0, (list, tuple)) and len(v0) >= 1: v0=v0[0]
                        if isinstance(v1, (list, tuple)) and len(v1) >= 1: v1=v1[0]
                        if isinstance(v0, int) and isinstance(v1, int):
                            side_diff += abs(((v0>>16)&0xFF)+((v0>>8)&0xFF)+(v0&0xFF) - ((v1>>16)&0xFF)-((v1>>8)&0xFF)-(v1&0xFF))
                    continuity = max(0, 1.0 - abs(diff_sum-side_diff)/max(diff_sum,1))

                    # 上方颜色
                    above_r=above_g=above_b=cnt2=0
                    for sr in range(row-20, row-5, 4):
                        for sc in range(area_cx-area_w//2, area_cx+area_w//2, 16):
                            val=img.get_pixel(sc,sr)
                            if isinstance(val,(list,tuple)) and len(val)>=1: v=val[0]
                            elif isinstance(val,int): v=val
                            else: continue
                            above_r+=(v>>16)&0xFF; above_g+=(v>>8)&0xFF; above_b+=v&0xFF; cnt2+=1
                    if cnt2>3: above_r//=cnt2; above_g//=cnt2; above_b//=cnt2
                    else: above_r=above_g=above_b=0

                    # 上下方颜色对比度（0~1）
                    if below_colors and above_r+above_g+above_b>0:
                        br,bg,bb=below_colors[0]
                        diff_ab=abs(br-above_r)+abs(bg-above_g)+abs(bb-above_b)
                        contrast=min(1.0, diff_ab/max(br+bg+bb, above_r+above_g+above_b, 1))
                    else:
                        contrast=0.5

                    # 动态权重：对比度低→梯度权重大；对比度高→颜色权重大
                    # 渐变方向：越往下越接近基准色 = 真液面
                    gradient_bonus = 0
                    if len(below_colors) >= 2:
                        d1 = ((below_colors[0][0]-base_r)**2+(below_colors[0][1]-base_g)**2+(below_colors[0][2]-base_b)**2)**0.5
                        dl = ((below_colors[-1][0]-base_r)**2+(below_colors[-1][1]-base_g)**2+(below_colors[-1][2]-base_b)**2)**0.5
                        gradient_bonus = max(0, (d1 - dl) / max(d1, 1)) * 2.0

                    # 权重重要性: ①下方颜色 ②渐变方向 ③梯度 ④一致性 ⑤连续性
                    # 重要性重新分配：下方颜色 > 渐变方向 > 梯度 > 一致性 > 连续性
                    w_grad = 0.15 + (1.0-contrast)*1.2
                    w_col  = 1.2 + contrast*5.0
                    w_con  = 0.3 + contrast*1.0

                    score = below_score*220*w_col + gradient_bonus*180 + diff_sum*40*w_grad + consistency*45*w_con + continuity*6
                    if score > best_score:
                        second_score = best_score
                        second_row = best_row
                        second_contrast = best_contrast
                        best_score = score
                        best_row = row
                        best_contrast = contrast
                    elif score > second_score:
                        second_score = score
                        second_row = row
                        second_contrast = contrast

                # 如果第二高的候选略低但更靠下且颜色变化更大，给偏下者加分
                if second_score > 0 and second_row > best_row:
                    if second_score >= best_score * 0.70 and second_contrast > 0.50:
                        boosted = second_score + best_score * 0.25
                        if boosted > best_score:
                            best_score = boosted
                            best_row = second_row
                            best_contrast = second_contrast
                if best_score > 10:
                    # 真实液面略低：高对比时向下微调 2px
                    offset = 2 if best_contrast > 0.6 else 0
                    level_y = min(best_row + offset, y + h - 1)


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

        # ===== 基准色输出（与色块检测同源） =====
        if _base_valid:
            hex_color = f"#{base_r:02X}{base_g:02X}{base_b:02X}"
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