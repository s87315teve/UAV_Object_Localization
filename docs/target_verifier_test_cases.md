# Target Verifier Test Cases

這份文件說明如何產生與測試「一般車、目標車、混淆車」同畫面的資料增強案例。

目前初賽 verifier 採用較寬鬆的規則：先確認候選框是白車，再確認車上能看到紅色或粉紅色標記。白色叉叉形狀只作為資料增強外觀，不再作為硬性判斷條件。

新腳本：

```bash
scripts/generate_target_verifier_test_data.py
scripts/generate_video03_target_test_video.py
scripts/opencv_test.py
scripts/run_augmented_localization_batch.py
```

輸出資料夾預設為：

```text
target_verifier_test_data/
├── images/
├── labels/
├── metadata.csv
├── metadata.json
└── objects.csv
```

每張圖都會包含：

- `ordinary`：一般車，顏色不限，不應被 target verifier 當成目標。
- `target`：白色目標車，直接使用航拍畫面中真實白色轎車 crop，保留原本車窗、陰影與車身紋理，只在車體上疊紅色或粉紅色底的標記，叉叉本身是白色。
- `confuser`：混淆車，例如白車無 X、紅車無 X、白車紅色塊、非白車帶 X。

## OpenCV 影片輸入流程

比賽展示流程建議用 `scripts/opencv_test.py` 測試。它會開 OpenCV 視窗讀 `stream.sdp`、影片檔、webcam 或 URL，同時錄影、定期存 frame；按視窗上的 `Detect` 按鈕時，才會把當下 frame 送進 `scripts/localize_vehicles.py` 做 YOLO + target verifier。

建議測試順序：

1. 用 `scripts/generate_video03_target_test_video.py` 從 `raw_videos/video03.MP4` 產生低畫質測試影片，並在每個輸出 frame 貼上一台目標白車。
2. 用 `scripts/opencv_test.py` 讀這支測試影片，確認視窗、錄影、按鈕觸發辨識、輸出檔都正常。
3. 改成 `--source stream.sdp` 或正式串流 URL，測比賽實際輸入來源。

影片產生器會直接引用 `scripts/generate_target_verifier_test_data.py` 裡的 `build_car_alpha` 與 `draw_target_marker`，所以白車 crop 去背方式、紅底白色粗叉叉生成方式，和影像資料增強測試集一致。

### 產生帶目標車的測試影片

推薦先用這組參數產生一支小於 30 MB、解析度 1280 px 寬、12 fps 的低畫質測試影片：

```bash
conda run -n uav_contest_env python scripts/generate_video03_target_test_video.py \
  --input-video raw_videos/video03.MP4 \
  --output-video augmented_test_data/videos/video03_target_low_quality.mp4 \
  --target-car vehicle_localization_outputs/frame_000051/crops/veh_001.jpg \
  --output-width 1280 \
  --fps 12 \
  --bitrate 900k \
  --max-size-mb 30 \
  --target-long-side 59 \
  --jpeg-quality 34 \
  --noise-sigma 5.0 \
  --blur-radius 1.1 \
  --seed 20260705 \
  --overwrite
```

輸出會包含：

```text
augmented_test_data/videos/video03_target_low_quality.mp4
augmented_test_data/videos/video03_target_low_quality.metadata.json
augmented_test_data/videos/video03_target_low_quality.preview.jpg
```

`metadata.json` 會記錄每個輸出 frame 對應的原影片 timestamp 與 `target_bbox_xyxy`，可用來檢查 verifier 是否抓到正確目標。

`--target-long-side 59` 會讓貼上的目標車長寬約為舊版 `118` 設定的一半，比較接近遠距離小車測試。

### 用 OpenCV 測試影片輸入與手動辨識

先用上一步產生的測試影片跑 `opencv_test.py`。OpenCV 視窗會用可調整模式開啟，下面指令會先開成 `1600 x 900`，之後仍可手動拉伸。Apple Silicon 用 `--localize-device mps`，NVIDIA GPU 改成 `--localize-device cuda:0`：

```bash
conda activate uav_contest_env
python scripts/opencv_test.py \
  --source augmented_test_data/videos/video03_target_low_quality.mp4 \
  --backend ffmpeg \
  --loop-source \
  --window-width 1600 \
  --window-height 900 \
  --output-root stream_outputs/video03_target_opencv_test \
  --frame-interval 1 \
  --record-segment-seconds 120 \
  --show-detection-result \
  --localize-device cuda:0 \
  --localize-model yolo26x.pt \
  --localize-vehicle-classes car \
  --localize-imgsz 1600 \
  --localize-tile-upscales 1,4 \
  --localize-yolo-batch-size 16 \
  --localize-conf 0.25 \
  --target-verifier \
  --target-verifier-min-score 0.18 \
  --target-verifier-min-white-ratio 0.15 \
  --target-verifier-min-red-pixels 35
```

OpenCV 視窗上有三個按鈕：

- `Save Frame`：把目前畫面存到 `<output-root>/manual_frames/`。
- `Detect`：存下目前畫面，並在背景執行一次 `localize_vehicles.py`；完成後會自動顯示 `03_process_overview.jpg`。
- `Quit`：關閉視窗；也可以按 `Esc`。

辨識輸出會在：

```text
stream_outputs/video03_target_opencv_test/detections/<detect_frame_name>/
├── 01_frame_vehicle_detections.jpg
├── 02_map_vehicle_coordinates.jpg
├── 03_process_overview.jpg
├── vehicle_localization.json
└── vehicle_localization.csv
```

常用參數效果：

| 參數 | 效果 |
| --- | --- |
| `--source` | OpenCV 影片來源，可用本機影片、`stream.sdp`、`0` webcam 或 RTSP/UDP/HTTP URL。 |
| `--backend` | OpenCV 讀取後端；影片檔、SDP、RTSP/UDP 通常用 `ffmpeg`，webcam 用 `default`。 |
| `--loop-source` | 只對本機影片檔有效；影片讀到結尾後自動從頭重播，直到按 `Quit`、`Esc` 或 `Ctrl-C`。 |
| `--window-width` / `--window-height` | OpenCV 主視窗初始大小；視窗使用可調整模式，開啟後仍可手動拉伸。 |
| `--output-root` | 錄影、定期 frames、手動 frames、Detect 辨識結果的根目錄。 |
| `--frame-interval` | 每隔幾秒自動存一張 frame；設 `0` 可關閉自動存圖。 |
| `--record` / `--no-record` | 是否將來源畫面錄成影片存到 `<output-root>/recordings/`；預設關閉，需要錄影時才加 `--record`。 |
| `--record-segment-seconds` | 錄影每幾秒切一個新檔；可降低單檔過大或中斷時的損失。 |
| `--show-detection-result` / `--no-show-detection-result` | Detect 完成後是否自動開另一個 OpenCV 視窗顯示最新辨識結果；預設開啟。 |
| `--result-window-name` | Detect 結果視窗名稱；預設是 `UAV localization result`。 |
| `--localize-device` | 傳給 YOLO 的推論裝置；Apple Silicon 用 `mps`，NVIDIA GPU 用 `cuda:0`，CPU 用 `cpu`。 |
| `--localize-model` | 傳給 `localize_vehicles.py` 的 YOLO 權重檔。 |
| `--localize-vehicle-classes` | YOLO 保留的類別；比賽目標車目前建議先用 `car`。 |
| `--localize-imgsz` | YOLO 輸入尺寸；越大越容易抓小車，但速度較慢、較吃記憶體。 |
| `--localize-tile-upscales` | tile 放大倍率；`1,2` 會同時跑原倍率與 2 倍放大，提高小車召回率但較慢。 |
| `--localize-yolo-batch-size` | 每次送進 YOLO 的 tile 數；GPU 記憶體夠可調大，爆記憶體時調小。 |
| `--localize-conf` | YOLO confidence 門檻；低一點比較不漏小車，再交給 target verifier 過濾。 |
| `--target-verifier` | Detect 時啟用白車 + 紅/粉紅標記過濾，只留下疑似目標車。 |
| `--target-verifier-min-score` | 紅/粉紅標記的綜合分數門檻；越高越嚴格。 |
| `--target-verifier-min-white-ratio` | 車框內白色比例門檻；越高越要求白車，可降低非白車誤判。 |
| `--target-verifier-min-red-pixels` | 車框內紅/粉紅 marker 最少像素數；越高越抗雜訊，但遠距離小 marker 可能被濾掉。 |

### 用比賽串流測試

如果正式來源是 repo 內的 `stream.sdp`，用同一組辨識參數，只換 `--source` 和 `--output-root`：

```bash
conda activate uav_contest_env
python scripts/opencv_test.py \
  --source stream.sdp \
  --backend ffmpeg \
  --window-width 1600 \
  --window-height 900 \
  --output-root stream_outputs/contest_stream_test \
  --frame-interval 2 \
  --record \
  --record-segment-seconds 120 \
  --show-detection-result \
  --localize-device mps \
  --localize-model yolo26x.pt \
  --localize-vehicle-classes car \
  --localize-imgsz 1280 \
  --localize-tile-upscales 1,2 \
  --localize-yolo-batch-size 16 \
  --localize-conf 0.12 \
  --target-verifier \
  --target-verifier-min-score 0.18 \
  --target-verifier-min-white-ratio 0.15 \
  --target-verifier-min-red-pixels 35
```

其他來源：

- webcam：`--source 0 --backend default`
- RTSP/UDP/HTTP URL：`--source "<URL>" --backend ffmpeg`

### 離線批次補充測試

如果要看很多 frames 的整體統計，可以額外走抽 frame + batch。這適合回歸測試，不是比賽展示主流程：

```bash
conda run -n uav_contest_env python scripts/extract_frames.py \
  --input-dir augmented_test_data/videos \
  --output-dir augmented_test_data/video03_target_frames \
  --interval 1 \
  --prefix video03_target \
  --overwrite

conda run -n uav_contest_env python scripts/run_augmented_localization_batch.py \
  --input-dir augmented_test_data/video03_target_frames \
  --output-root augmented_test_data/video03_target_localization_outputs \
  --detector yolo \
  --yolo-model yolo26x.pt \
  --device mps \
  --vehicle-classes car \
  --yolo-batch-size 4 \
  --imgsz 1280 \
  --tile-upscales 1,2 \
  --target-verifier \
  --target-verifier-min-score 0.18 \
  --target-verifier-min-white-ratio 0.15 \
  --target-verifier-min-red-pixels 35 \
  --orientations all \
  --match-workers 4 \
  --feature-max-dim 1200 \
  --overwrite
```

批次摘要在：

```text
augmented_test_data/video03_target_localization_outputs/summary.csv
augmented_test_data/video03_target_localization_outputs/summary.json
```

### 拖曳 Frame GUI 單張辨識

`scripts/localize_frame_gui.py` 可以用 GUI 測單張 frame。把圖片拖進視窗，或按 `Open Frame` 選圖；按 `Run Detect` 後會執行 `scripts/localize_vehicles.py`，完成後直接在 GUI 顯示 `03_process_overview.jpg`。

拖曳功能需要 `tkinterdnd2`。已在 `requirements.txt` 中列出；如果目前環境還沒安裝，先跑：

```bash
conda run -n uav_contest_env python -m pip install -r requirements.txt
```

用目前推薦 CUDA 參數開 GUI：

```bash
conda run -n uav_contest_env python scripts/localize_frame_gui.py \
  --output-root frame_gui_outputs \
  --window-width 1600 \
  --window-height 950 \
  --detector yolo \
  --localize-device cuda:0 \
  --localize-model yolo26x.pt \
  --localize-vehicle-classes car \
  --localize-imgsz 1600 \
  --localize-tile-upscales 1,4 \
  --localize-yolo-batch-size 16 \
  --localize-conf 0.25 \
  --orientations all \
  --match-workers 4 \
  --feature-max-dim 1200 \
  --target-verifier \
  --target-verifier-min-score 0.18 \
  --target-verifier-min-white-ratio 0.15 \
  --target-verifier-min-red-pixels 35
```

也可以啟動時直接載入一張圖：

```bash
conda run -n uav_contest_env python scripts/localize_frame_gui.py \
  --frame path/to/frame.jpg \
  --localize-device cuda:0 \
  --localize-imgsz 1600 \
  --localize-tile-upscales 1,4 \
  --localize-yolo-batch-size 16 \
  --localize-conf 0.25 \
  --target-verifier
```

GUI 會把每次結果寫到 `--output-root/<frame_stem>_<timestamp>/`，包含 `01_frame_vehicle_detections.jpg`、`02_map_vehicle_coordinates.jpg`、`03_process_overview.jpg`、`vehicle_localization.json`、`vehicle_localization.csv` 和 `run.stdout.log` / `run.stderr.log`。

## 產生測試資料

使用預設設定產生 24 張：

```bash
conda activate uav_contest_env
python scripts/generate_target_verifier_test_data.py \
  --output-dir target_verifier_test_data \
  --count 24 \
  --overwrite
```

產生較小 smoke-test 資料集：

```bash
conda activate uav_contest_env
python scripts/generate_target_verifier_test_data.py \
  --output-dir target_verifier_test_data_smoke \
  --count 6 \
  --video-frame-count 2 \
  --ordinary-cars 2 \
  --target-cars 1 \
  --confuser-cars 2 \
  --overwrite
```

產生較難的 verifier 壓力測試：

```bash
conda activate uav_contest_env
python scripts/generate_target_verifier_test_data.py \
  --output-dir target_verifier_test_data_hard \
  --count 60 \
  --ordinary-cars 3 \
  --target-cars 1 \
  --confuser-cars 4 \
  --marker-size-min 0.10 \
  --marker-size-max 0.22 \
  --seed 20260706 \
  --overwrite
```

## 參數怎麼改

| 參數 | 效果 |
| --- | --- |
| `--count` | 產生幾張影像。越多越適合穩定度測試，但批次定位會跑比較久。 |
| `--ordinary-cars` | 每張圖的一般車數量。增加後可以測 verifier 是否誤收普通車。 |
| `--target-cars` | 每張圖的目標車數量。初賽建議先用 `1`，符合「先找到一個可靠目標」策略。 |
| `--confuser-cars` | 每張圖的混淆車數量。增加後可以測白車無 X、紅車、紅色塊、非白車 X 的誤判率。 |
| `--marker-size-min` / `--marker-size-max` | X 標記相對車身的尺寸範圍。調小會更難偵測；調大會比較容易。 |
| `--video-frame-count` | 從 `raw_videos/` 抽幾張背景。設 `0` 時只用 repo 裡現有測試圖和已抽出的紅叉範例。 |
| `--seed` | 固定亂數種子。相同 seed 會產生相同案例，方便比較不同 verifier 設定。 |
| `--overwrite` | 覆蓋既有輸出資料夾。沒有加時，資料夾已存在會停止，避免誤刪舊結果。 |

target 車目前會優先使用：

```text
vehicle_localization_outputs/frame_000051/crops/veh_001.jpg
vehicle_localization_outputs/frame_000051/crops/veh_002.jpg
```

這兩張是真實航拍白車 crop；資料增強只會把 marker 疊上去，不會再把任意車強制染成白車。

## 輸出欄位

`metadata.csv` 是每張影像一列，包含來源背景、物件數量、增強操作與 seed。

`objects.csv` 是每個物件一列，重點欄位：

| 欄位 | 說明 |
| --- | --- |
| `filename` | 對應的影像檔名。 |
| `role` | `ordinary`、`target` 或 `confuser`。 |
| `bbox_xyxy` | 車輛框，格式為 `[x1, y1, x2, y2]`。 |
| `marker_bbox_xyxy` | 有畫 X 或紅色混淆塊時才會有值；marker 會被裁切在車體 alpha 內，不應落到地面。 |
| `style` | 車輛顏色或混淆類型。 |
| `marker_style` | X 標記尺寸、紅色底色、白色叉叉顏色、旋轉後車身 scale/angle。 |

每張圖也有一份獨立 JSON label：

```text
target_verifier_test_data/labels/target_case_000001.json
```

## 單張測試

用 YOLO + target verifier 跑單張，這是比較接近正式 workflow 的測法：

```bash
conda activate uav_contest_env
python scripts/localize_vehicles.py \
  --frame target_verifier_test_data/images/target_case_000001.jpg \
  --output-dir target_verifier_test_outputs/target_case_000001 \
  --yolo-model yolo26x.pt \
  --device mps \
  --vehicle-classes car \
  --yolo-batch-size 4 \
  --imgsz 1280 \
  --tile-upscales 1,2 \
  --target-verifier \
  --target-verifier-context 0 \
  --target-verifier-min-score 0.25 \
  --target-verifier-min-white-ratio 0.20 \
  --target-verifier-min-red-pixels 120 \
  --orientations all \
  --match-workers 4 \
  --feature-max-dim 1200
```

快速檢查 verifier 與輸出格式，不想等 YOLO 時可以用白色 heuristic：

```bash
conda activate uav_contest_env
python scripts/localize_vehicles.py \
  --frame target_verifier_test_data/images/target_case_000001.jpg \
  --output-dir target_verifier_test_outputs/target_case_000001_fast \
  --detector white-heuristic \
  --heuristic-min-area 40 \
  --heuristic-max-area 50000 \
  --target-verifier \
  --orientations none \
  --feature-max-dim 900
```

注意：`white-heuristic` 只適合 smoke test，不能代表正式 YOLO 偵測準確度。

## 整批測試

用 YOLO 跑整批：

```bash
conda activate uav_contest_env
python scripts/run_augmented_localization_batch.py \
  --input-dir target_verifier_test_data/images \
  --output-root target_verifier_test_outputs_yolo \
  --yolo-model yolo26x.pt \
  --device mps \
  --vehicle-classes car \
  --yolo-batch-size 4 \
  --imgsz 1280 \
  --tile-upscales 1,2 \
  --target-verifier \
  --target-verifier-min-score 0.25 \
  --target-verifier-min-white-ratio 0.20 \
  --target-verifier-min-red-pixels 120 \
  --orientations all \
  --match-workers 4 \
  --feature-max-dim 1200 \
  --overwrite
```

快速 smoke test：

```bash
conda activate uav_contest_env
python scripts/run_augmented_localization_batch.py \
  --input-dir target_verifier_test_data/images \
  --output-root target_verifier_test_outputs_fast \
  --detector white-heuristic \
  --target-verifier \
  --target-verifier-min-white-ratio 0.35 \
  --orientations none \
  --match-workers 1 \
  --feature-max-dim 900 \
  --overwrite
```

看 summary：

```bash
python - <<'PY'
import csv
from pathlib import Path

path = Path("target_verifier_test_outputs_yolo/summary.csv")
with path.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        print(
            row["filename"],
            "detections=" + row["detections_count"],
            "kept=" + row["target_verifier_kept"],
            "rejected=" + row["target_verifier_rejected"],
            "map=" + row["map_method"],
            "time=" + row["timing_total"],
        )
PY
```

## Verifier 調參

如果一般車或混淆車太常被留下，讓 verifier 更嚴格：

```bash
--target-verifier-context 0
--target-verifier-min-score 0.25
--target-verifier-min-white-ratio 0.20
--target-verifier-min-red-pixels 120
```

如果目標車常被濾掉，讓 verifier 更寬鬆：

```bash
--target-verifier-min-score 0.12
--target-verifier-min-white-ratio 0.15
--target-verifier-min-red-pixels 20
```

調參效果：

- `min-score` 越高，紅色標記需要越明顯、越集中。
- `min-white-ratio` 越高，越要求車框本身是白車，可降低紅車或非白車 X 誤判。
- `min-red-pixels` 越高，越不容易被小紅點或雜訊騙過，但遠距離小標記也可能被濾掉。

## 建議測試順序

1. 先產生 6 張 smoke test。
2. 用 `white-heuristic` 確認腳本與輸出欄位正常。
3. 用 YOLO 跑 1 張看視覺輸出。
4. 用 YOLO 跑整批，讀 `summary.csv`。
5. 如果誤收混淆車，先提高 `min-white-ratio`，再提高 `min-red-pixels`。
6. 如果漏掉目標車，先降低 `min-red-pixels`，再降低 `min-score`。
