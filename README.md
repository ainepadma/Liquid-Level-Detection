# Liquid-Level-Detection
Problem D of the 28th Electronics Design Contest of Southeast University : A beverage cup measuring device based on monocular vision

2026年东南大学第28届电子设计竞赛D题：基于单目视觉的饮料杯测量装置


For further details please contact: Ainepadma24@gmail.com

## Project Construction

### Hardware List

1. Sipeed MaixCAM PRO RISC-V
2. Arduino 0.96 Inch OLED screen (SSD1315, I2C, 128×64)
3. STM32F103C8T6 system board (ARM Cortex-M3)
4. BWLBVR 100 milliohm precision resistor
5. JY-SDM12 M12 5 MegaPixel 6.0-22mm 650nm lens (2.8-12mm version is more suitable for this problem)
6. INA240A1 current sampling module
7. INA240A1PWR TSSOP-8 chip
8. 830-hole breadboard
9. 9×15 peg board
10. A lot of Dupont wires
11. Hot melt glue gun and soldering tin

### Compatible Software

1. Microsoft Visual Studio Code
2. STM32CubeIDE
3. MaixVision (Sipeed official IDE for MaixCAM series)



## Project Instruction

### 1. Overview

This project implements a **monocular vision-based beverage cup measuring device**. A single camera (GC4653 sensor, 2560×1440) captures the cup from a top-down or side view, and a dual-model YOLOv5 pipeline detects the cup/bottle, estimates the liquid surface position, and computes the liquid height relative to the cup bottom. The measured parameters are transmitted to an STM32F103C8T6 MCU via UART for further processing and displayed on a 0.96-inch SSD1315 I2C OLED screen.

### 2. System Architecture

```
┌──────────────────────────────────────────────────────┐
│                    MaixCAM PRO                       │
│  ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ GC4653   │──▶│ YOLOv5   │──▶│ Liquid Level   │  │
│  │ 2560×1440│   │ Dual     │   │ Detection      │  │
│  │ Camera   │   │ Models   │   │ + Ref.Color    │  │
│  └──────────┘   └──────────┘   └───────┬────────┘  │
│                                        │ UART       │
└────────────────────────────────────────┼────────────┘
                                         │ /dev/ttyS0
                                         ▼
┌──────────────────────────────────────────────────────┐
│                STM32F103C8T6 MCU                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐│
│  │ UART RX  │──▶│ Data     │──▶│ SSD1315 OLED     ││
│  │ (Vision) │   │ Fusion   │   │ 0.96" 128×64     ││
│  └──────────┘   └────┬─────┘   └──────────────────┘│
│                      │ ▲                             │
│  ┌──────────────────┐│ │                             │
│  │ INA240A1 Current │──▶│                             │
│  │ Sampling Module  │  │                             │
│  │ (100mΩ shunt)    │  │                             │
│  └──────────────────┘  │                             │
└──────────────────────────────────────────────────────┘
```

**Data flow**:
1. MaixCAM PRO captures 2560×1440 RGB frames via GC4653 MIPI sensor.
2. Two YOLOv5 models (custom 2-class model + COCO-pretrained model) run inference in parallel.
3. Results are merged, filtered by confidence and size constraints, and the best candidate is locked via IoU-based tracking with EMA smoothing.
4. Liquid surface is detected inside the object bounding box using blob detection and gradient scoring.
5. Reference color is sampled from the region between the liquid surface and the object bottom.
6. Measurement data (height, top margin, bottom margin, confidence, average color, liquid level distance) is sent every 3 frames to STM32 via UART at 115200 baud.
7. **Current detection**: INA240A1 module samples voltage across a 100 mΩ precision shunt resistor; the STM32 ADC reads the amplified signal to compute real-time current.
8. STM32 fuses vision data and current data, then displays object/liquid/current status on the SSD1315 OLED.

### 3. Key Algorithms

#### 3.1 Dual-Model YOLOv5 Detection

Two `.cvimodel` models run on the CV181x NPU:

| Model | Path | Labels | Input Size |
|-------|------|--------|------------|
| Custom (A) | `/root/models/model_274449.cvimodel` | `['cup', 'bottle']` | 320×320 |
| COCO (B) | `/root/models/yolov5s_320x224_int8.cvimodel` | 80-class (IDs 39=bottle, 41=cup) | 320×224 |

Model A provides higher precision for the target classes; Model B provides a lower-confidence fallback (scaled by `DETECT_B_CONF_SCALE`). Candidates from both models are merged, scored by a weighted sum of confidence, center proximity, previous-frame proximity, and liquid-level proximity, then filtered by per-class minimum confidence thresholds and size constraints.

#### 3.2 Object Tracking & Locking

- **IoU-based matching**: the best candidate is matched against the locked box from the previous frame.
- **EMA smoothing**: when locked, object coordinates are smoothed with exponential moving average (α=0.7) to reduce jitter.
- **Delayed unlock**: a miss counter allows up to `LOCK_MISS_MAX` consecutive frames without a good match before releasing the lock.
- **Color-based verification**: periodically samples pixels inside the locked box; if the color deviates significantly, the lock is released.

#### 3.3 Liquid Surface Detection

Inside the object's bounding box lower half (ROI: `y + h/3` to `y + 5h/6`):
1. **Blob detection**: `find_blobs` with dark-pixel thresholds locates the rough liquid region.
2. **Candidate scoring**: each row in the scan range is scored by:
   - Vertical gradient (edge strength) at 3 horizontal positions.
   - Color similarity of 3 sampling patches below the candidate row to the reference base color.
   - Color consistency among the 3 below-patch samples.
   - Above/below contrast.
   - Gradient direction bonus (colors approaching the reference color as depth increases = true liquid surface).
3. **Secondary candidate boost**: if the second-best row is lower and has high contrast, it may override the primary candidate.
4. **Locking**: once the liquid surface stabilizes (8 consecutive frames within ±15 px), the Y coordinate is locked. If the detected surface drifts outside the lock window, the lock is released.

#### 3.4 Reference Color Extraction

The reference ("base") color is obtained by sampling the region **between the previous-frame liquid surface estimate and the object bottom**. Pixels are sampled in a central column strip (`w/3` wide), then averaged and smoothed with EMA (α=0.15). This reference color is used by the liquid surface scoring algorithm to distinguish liquid from cup body.

#### 3.5 Edge Refinement

After YOLO detection, the top and bottom edges of the object bounding box are refined using gradient+color scoring along the vertical midline:
- **Top edge**: gradient + color distance to the background above the object.
- **Bottom edge**: gradient + color distance to a sampled bottom reference color.

If the refined score exceeds a threshold, the edge coordinate is adjusted.

### 4. UART Communication Protocol

- **Baud rate**: 115200, 8N1
- **Port**: `/dev/ttyS0` (UART0) on MaixCAM; received by STM32 UART.
- **Output format** (CSV line every 3 frames):
  ```
  avg_height,avg_top,avg_bottom,avg_confidence,avg_color_hex,avg_level_distance
  ```
  | Field | Description | Unit |
  |-------|-------------|------|
  | `avg_height` | Object bounding box height | px |
  | `avg_top` | Top margin (Y coordinate) | px |
  | `avg_bottom` | Bottom margin (CAM_H - bottom Y) | px |
  | `avg_confidence` | Detection confidence | 0–1 |
  | `avg_color` | Average color inside box | RRGGBB hex |
  | `avg_level_distance` | Liquid surface to cup bottom distance | px (−1 if no liquid detected) |

- In **stable state** (object position change < 2% for ≥15 frames), median values are output instead of means for robustness against outliers.

### 5. Key Tuning Parameters

All tunable parameters are declared at the top of `launch/main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DETECT_CONF_TH` | 0.22 | Model A base confidence threshold |
| `DETECT_IOU_TH` | 0.45 | NMS IoU threshold |
| `DETECT_B_CONF_SCALE` | 0.50 | Model B threshold multiplier |
| `CUP_CONF_MIN` | 0.15 | Minimum confidence for cup |
| `BOTTLE_CONF_MIN` | 0.22 | Minimum confidence for bottle |
| `LOCK_IOU_MIN` | 0.30 | IoU threshold for object locking |
| `LOCK_MISS_MAX` | 6 | Max consecutive misses before unlock |
| `LOCK_EMA_ALPHA` | 0.7 | EMA smoothing weight for locked position |
| `CAND_SCORE_W` | 5.0 | Candidate score weight: confidence |
| `CAND_CENTER_W` | 2.0 | Candidate score weight: center proximity |
| `CAND_PREV_W` | 4.0 | Candidate score weight: previous-frame proximity |
| `CAND_LEVEL_W` | 3.0 | Candidate score weight: liquid-level proximity |
| `LEVEL_PROX_SCALE` | 200.0 | Liquid proximity distance normalization (px) |
| `LEVEL_EMA_ALPHA` | 0.45 | EMA weight for liquid level estimate update |

### 6. File Structure

```
MaixVision/
├── launch/
│   ├── main.py                     # Main application (current version)
│   ├── app.yaml                    # MaixVision app configuration
│   ├── maix-liquid_detection-v1.0.1/
│   ├── maix-liquid_detection-v1.0.2/
│   ├── maix-liquid_detection-v1.0.3/
│   ├── maix-liquid_detection-v1.0.4/
│   ├── maix-liquid_detection-v1.0.5/
│   ├── maix-liquid_detection-v1.0.6/
│   ├── maix-liquid_detection-v1.0.7/
│   ├── maix-liquid_detection-v1.0.8/
│   ├── maix-liquid_detection-v1.1.1/
│   ├── maix-liquid_detection-v1.1.3/
│   ├── maix-liquid_detection-v1.2.1/
│   ├── maix-liquid_detection-v1.3.2/
│   ├── maix-liquid_detection-v1.3.3/
│   ├── maix-liquid_detection-v1.3.4/
│   ├── maix-liquid_detection-v1.3.5/ (our final vesion)
│   └── maix-liquid_detection-v1.3.6/ (the latest vesion)
└── backup/
    ├── 0317.py                     # Early prototype (March 17)
    ├── 120.py                      # Variant with different detection params
    ├── finished.py                 # Feature-complete snapshot
    ├── old_model.py                # Single-model variant (reference)
    └── old_model_color.py          # Color-enhanced variant (reference)


Models/
├── dataset/                    # Training dataset for custom model
├── model_273912.maixcam        # Custom 2-class model package
├── model_274449.maixcam        # Custom 2-class model package (the latest and our final vesion)
└── yolov5s.maixcam             # COCO YOLOv5s model package


STM32/
├── luanch/
│   └── vesion01/ 
└── backup/
    └── test.c                  # STM32 test routines


README.md

```

### 7. Build & Deployment

#### MaixCAM (Python)
1. Open the project in **MaixVision IDE**.
2. Connect MaixCAM PRO via USB.
3. Select `launch/main.py` as the entry point.
4. Click **Run** — the script is uploaded to `/tmp/maixpy_run/main.py` and executed.
5. The 480×640 display shows detection boxes, liquid line, and status indicators.

#### STM32 (C)
1. Open the STM32 project in **STM32CubeIDE**.
2. Build with `gcc-arm-none-eabi` toolchain.
3. Flash via **ST-Link v2** debugger.
4. The MCU reads UART data, parses CSV lines, and updates the SSD1315 OLED.

### 8. Notes

- **Model load order**: `Camera` must be initialized before `nn.YOLOv5` to avoid shared memory allocation conflicts with the 2560×1440 VI pool (~16.6 MB).
- **Display**: call `img.resize(disp.width(), disp.height())` before `disp.show()` to avoid `enc_jpeg_in` VB pool overflow.
- **High resolution**: the GC4653 sensor natively captures 2560×1440 at 25 fps. The YOLO models run at 320×224/320×320 input sizes; the NPU handles rescaling internally.
- **Lens recommendation**: the 2.8–12mm variant of the JY-SDM12 lens provides better field-of-view coverage for nearby cups (typical working distance 20–50 cm).

### 9. Acknowledgments

- **Sipeed** — MaixCAM PRO hardware and MaixPy SDK
- **Southeast University** — 28th Electronics Design Contest organization
- **GitHub Copilot** — AI-assisted development (DeepSeek V4 Pro / GPT-5.2-Codex) (Cumulative token usage: 400–500 million)

---

*Project completed May 2026.*
