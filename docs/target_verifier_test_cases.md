# Target Verifier Test Cases

這份文件說明如何產生與測試「一般車、目標車、混淆車」同畫面的資料增強案例。

目前初賽 verifier 採用較寬鬆的規則：先確認候選框是白車，再確認車上能看到紅色或粉紅色標記。白色叉叉形狀只作為資料增強外觀，不再作為硬性判斷條件。

新腳本：

```bash
scripts/generate_target_verifier_test_data.py
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
  --target-verifier-context 0 \
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
