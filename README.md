[修改紀錄](MODIFICATION_LOG.md)

# UAV Object Localization

本專案目標是建立一套無人機地面車輛定位流程：使用無人機拍攝目標區域影像，辨識影像中的車輛，並把車輛在影像中的位置轉換成地圖上的 GPS 座標。

完整流程說明請見 [docs/drone_vehicle_geolocation_workflow.md](docs/drone_vehicle_geolocation_workflow.md)。
目標車 verifier 的資料增強與測試指令請見
[docs/target_verifier_test_cases.md](docs/target_verifier_test_cases.md)。
帶目標車的影片測試資料說明請見
[docs/video03_target_test_video.md](docs/video03_target_test_video.md)。

## 專案要解決的問題

核心任務是把「影像中的車輛位置」轉成「可顯示在地圖上的 GPS 位置」。做法不是只比對車輛本體，因為車輛會移動，參考地圖上不一定有同一台車；更穩定的方向是比對車輛周圍的道路標線、路面紋理、建物邊緣與固定地物，再推回車輛中心點的 GPS。

## 已知輸入與主要輸出

| 類別 | 內容 |
| --- | --- |
| 已知輸入 | 目標區域邊界 GPS、無人機影像、無人機 GPS/IMU、相機參數 |
| 主要輸出 | 車輛 ID、車輛中心點 GPS、信心分數、所在方格 ID |
| 核心方法 | 地圖切格、車輛偵測、影像配準、像素座標轉 GPS |
| 重要前提 | 相機時間、無人機姿態、GPS/IMU 與影像幀需要同步 |

## 預期工作流程

1. 定義目標區域 GPS 邊界。
2. 拍攝參考航拍影像，建立可查詢的地理參考地圖。
3. 將參考地圖切成 GPS 方格，建立 `grid_id`、四角 GPS、中心 GPS、像素範圍與影像特徵索引。
4. 任務飛行時拍攝影像，使用車輛偵測模型取得 bounding box 與車輛中心點。
5. 使用無人機 GPS/IMU 與相機模型先粗估候選位置。
6. 擷取車輛周圍地面區塊，和參考地圖候選方格做影像配準。
7. 將車輛像素位置轉換為 GPS，融合投影與匹配結果後輸出信心分數。
8. 在地圖介面顯示車輛位置，並匯出結構化偵測紀錄。

## 建議系統模組

| 模組 | 建議內容 | 目的 |
| --- | --- | --- |
| 車輛偵測 | YOLO 系列、RT-DETR 或 aerial-view detector | 找出影像中的車輛框 |
| 影像配準 | ORB、SIFT、SuperPoint、LoFTR 或特徵匹配加 RANSAC | 將任務影像對齊參考地圖 |
| 幾何投影 | 相機內參、外參、無人機高度、姿態角、地面平面假設 | 從影像像素推估地面座標 |
| 座標轉換 | 參考地圖 geo-transform、方格四角 GPS 內插 | 將地圖像素轉成 GPS |
| 信心評估 | 偵測分數、匹配分數、投影與匹配距離差 | 判斷定位結果是否可靠 |

## 最小可行版本

第一版建議先做離線版，縮小範圍並驗證定位誤差是否符合需求。

| 階段 | 必做功能 | 暫緩功能 |
| --- | --- | --- |
| MVP 1 | 離線航拍影像、人工建立參考圖、方格索引 | 即時串流 |
| MVP 2 | 車輛偵測、候選方格搜尋、粗略 GPS | 多車追蹤 |
| MVP 3 | 影像配準、像素級 GPS 內插、信心分數 | 完整自動飛行任務 |
| MVP 4 | 地圖顯示、結果匯出、誤差報告 | 邊緣端即時部署 |

## 建議輸出格式

每次偵測到車輛後，建議輸出結構化紀錄，方便後續地圖顯示、軌跡追蹤與誤差分析。

| 欄位 | 範例 | 說明 |
| --- | --- | --- |
| `timestamp` | `2026-06-23T10:15:30Z` | 影像時間 |
| `frame_id` | `flight_001_frame_0234` | 影像幀編號 |
| `vehicle_id` | `veh_0007` | 車輛追蹤 ID |
| `bbox_xyxy` | `[530, 210, 590, 260]` | 車輛框座標 |
| `grid_id` | `A03_B12` | 最接近的地圖方格 |
| `latitude` | `25.012345` | 車輛緯度 |
| `longitude` | `121.543210` | 車輛經度 |
| `confidence` | `0.87` | 綜合信心分數 |
| `method` | `projection_match_fusion` | 使用的定位方法 |

## 評估重點

| 評估項目 | 量測方式 |
| --- | --- |
| 偵測準確度 | precision、recall、mAP |
| 定位誤差 | 預測 GPS 與真值 GPS 的距離誤差 |
| 方格命中率 | 預測方格是否包含真實車輛位置 |
| 匹配穩定度 | 不同高度、角度、光照下的匹配成功率 |
| 即時性 | 每張影像處理時間與端到端延遲 |

## 目前資料

- `docs/drone_vehicle_geolocation_workflow.md`: 無人機車輛 GPS 定位完整工作流程與架構圖。
- `docs/images/`: workflow 文件使用的流程示意圖。
- `raw_videos/`: 原始無人機影片資料。

## UAV Telemetry UDP 傳輸

Nano 端可用 `scripts/uav_telemetry_udp_sender.py` 從 Pixhawk 讀取高度與電池電壓，打包成 JSON 後透過 UDP 傳到其他電腦。預設目標是 `192.168.1.150:6001`，JSON 欄位包含：

```json
{
  "timestamp": "2026-07-06T12:00:00.000Z",
  "sequence": 0,
  "altitude_m": -1.87,
  "relative_altitude_m": -3.63,
  "battery_voltage_v": 23.21
}
```

接收端電腦先執行：

```bash
python3 scripts/uav_telemetry_udp_receiver.py --host 0.0.0.0 --port 6001
```

Jetson Nano 端再執行：

```bash
python3 scripts/uav_telemetry_udp_sender.py \
  --serial /dev/ttyACM0 \
  --baud 115200 \
  --host 192.168.1.150 \
  --port 6001
```

如果接收端防火牆有開啟，需允許 UDP `6001`。若要調整傳送頻率，可加上 `--rate-hz 10`；設成 `--rate-hz 0` 則會在相關 MAVLink 訊息更新時盡量送出。
sender 會在 Pixhawk busy、heartbeat timeout、USB 中途斷線或一段時間沒有 MAVLink 訊息時自動關閉 serial 並重連。常用穩定性參數：

```bash
python3 -u scripts/uav_telemetry_udp_sender.py \
  --serial /dev/ttyACM0 \
  --baud 115200 \
  --host 192.168.1.150 \
  --port 6001 \
  --heartbeat-timeout 10 \
  --message-timeout 5 \
  --reconnect-delay 5
```

`scripts/opencv_test.py` 也會預設監聽 UDP `6001`，並在影像下方顯示高度、相對高度與電池電壓。沒有收到 telemetry 時會顯示 `None`，不會影響影片顯示或辨識流程。若要改 port，可加上 `--telemetry-port 6001`；若要關閉可加上 `--no-telemetry`。

本專案的 `stream.sdp` 影片串流使用 RTP `5000`，FFmpeg/OpenCV 可能同時佔用 RTCP `5001`，因此 telemetry 使用 `6001` 避免和影片串流衝突。

## 從原始影片抽取影像

`scripts/extract_frames.py` 會依照 `raw_videos/` 裡的影片檔名排序處理影片，預設每 3 秒擷取一張影像，並把三支影片的截圖依任務順序連續存到同一個資料夾。

執行前需要先安裝 `ffmpeg`，並確認 `ffmpeg` 可以在終端機直接執行。

```bash
python3 scripts/extract_frames.py
```

預設輸出資料夾是 `extracted_frames/`，檔名會是：

```text
frame_000001.jpg
frame_000002.jpg
frame_000003.jpg
...
```

若要指定輸入、輸出資料夾或抽圖間隔，可以使用：

```bash
python3 scripts/extract_frames.py \
  --input-dir raw_videos \
  --output-dir extracted_frames \
  --interval 3
```

如果輸出資料夾已經有檔案，腳本會停止以避免覆蓋或混入舊資料。確定要寫入既有資料夾時再加上 `--overwrite`。

## OpenCV 影片輸入與比賽流程測試

比賽展示時建議用 `scripts/opencv_test.py` 測整條影片輸入路徑。這支程式會開 OpenCV 視窗讀 `stream.sdp`、影片檔、webcam 或 URL，同時錄影、定期存 frame；按視窗上的 `Detect` 按鈕時，會把當下 frame 送進 `scripts/localize_vehicles.py` 做 YOLO + target verifier。

若要產生一支帶目標白車的低畫質測試影片，使用：

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

這支腳本會使用 `scripts/generate_target_verifier_test_data.py` 內相同的白車 crop alpha 方法與紅底白色粗叉叉 marker 生成方法。預設輸出：

```text
augmented_test_data/videos/video03_target_low_quality.mp4
augmented_test_data/videos/video03_target_low_quality.metadata.json
augmented_test_data/videos/video03_target_low_quality.preview.jpg
```

`--target-long-side 59` 會讓貼上的目標車長寬約為舊版 `118` 設定的一半，比較接近遠距離小車測試。

先用這支測試影片確認 OpenCV 視窗、錄影、手動觸發辨識都能 work。OpenCV 視窗會用可調整模式開啟，下面指令會先開成 `1600 x 900`，之後仍可手動拉伸。Apple Silicon 用 `--localize-device mps`，NVIDIA GPU 改成 `--localize-device cuda:0`：

```bash
conda run -n uav_contest_env python scripts/opencv_test.py \
  --source augmented_test_data/videos/video03_target_low_quality.mp4 \
  --backend ffmpeg \
  --loop-source \
  --window-width 1600 \
  --window-height 900 \
  --output-root stream_outputs/video03_target_opencv_test \
  --frame-interval 1 \
  --record \
  --record-segment-seconds 120 \
  --show-detection-result \
  --localize-device mps \
  --localize-model yolo26x.pt \
  --localize-vehicle-classes car \
  --localize-imgsz 1280 \
  --localize-tile-upscales 1,2 \
  --localize-yolo-batch-size 4 \
  --localize-conf 0.12 \
  --target-verifier \
  --target-verifier-min-score 0.18 \
  --target-verifier-min-white-ratio 0.15 \
  --target-verifier-min-red-pixels 35
```

視窗按鈕用途：

- `Save Frame`：把目前畫面存到 `<output-root>/manual_frames/`。
- `Detect`：存下目前畫面，並在背景執行一次定位辨識；完成後會自動顯示 `03_process_overview.jpg`。
- `Quit` 或 `Esc`：關閉視窗。

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

正式接比賽來源時，如果來源是 repo 內的 `stream.sdp`，用：

```bash
conda run -n uav_contest_env python scripts/opencv_test.py \
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
  --localize-yolo-batch-size 4 \
  --localize-conf 0.12 \
  --target-verifier \
  --target-verifier-min-score 0.18 \
  --target-verifier-min-white-ratio 0.15 \
  --target-verifier-min-red-pixels 35
```

如果是 webcam，來源可改成 `--source 0 --backend default`。如果是 RTSP/UDP/HTTP URL，直接把 URL 放在 `--source`，通常保留 `--backend ffmpeg`。

離線大量回歸測試仍可走抽 frame + batch。這不等於比賽即時流程，但適合看很多 frame 的 summary：

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

## 拖曳 Frame GUI 單張辨識

如果要挑單張 frame 快速測辨識，可以用 `scripts/localize_frame_gui.py`。GUI 會讓你把圖片拖進視窗，或按 `Open Frame` 選圖；按 `Run Detect` 後會執行 `scripts/localize_vehicles.py`，完成後直接在 GUI 顯示 `03_process_overview.jpg`。

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

## 將抽出的影像拼接成大圖

`scripts/stitch_frames.py` 會讀取 `extracted_frames/` 中現有的影像，依檔名排序後做特徵匹配與 RANSAC 對齊，再把所有 frame 投影到同一張大畫布。你可以先手動刪掉不想使用的頭尾 frame，腳本只會使用資料夾裡剩下的影像。

方法重點：

- 預設使用 SIFT 特徵做匹配；如果目前 OpenCV 不支援 SIFT，會退回 ORB。
- 相鄰 frame 會先估計 `homography`，再累積到第一張 frame 的座標系。
- 輸出不會被強制拉伸成長方形地圖；沒有影像覆蓋的區域會保留為全黑。
- 重疊區域使用 feather blending，降低接縫突兀感。
- `--cuda` 會改用 OpenCV CUDA ORB 特徵加速；這需要自行安裝支援 CUDA 的 OpenCV，`pip install opencv-python` 通常不包含 CUDA。

先安裝 Python 套件：

```bash
python3 -m pip install -r requirements.txt
```

使用預設設定拼接：

```bash
python3 scripts/stitch_frames.py
```

預設會讀取：

```text
extracted_frames/
```

並輸出：

```text
stitched_outputs/mosaic.png
```

如果要指定輸入和輸出：

```bash
python3 scripts/stitch_frames.py \
  --input-dir extracted_frames \
  --output stitched_outputs/mosaic.png
```

如果你的 OpenCV 是支援 CUDA 的版本，可以加上：

```bash
python3 scripts/stitch_frames.py --cuda
```

如果結果局部看起來有透視拉扯，可以改用較保守的 affine 模型：

```bash
python3 scripts/stitch_frames.py --transform affine
```

如果輸出檔已存在，腳本會停止以避免覆蓋；確定要覆蓋時加上 `--overwrite`。

### 固定航高垂直航帶拼接

如果無人機高度、角度固定，且路徑大致是垂直等速移動，建議先使用
`scripts/stitch_vertical_strip.py`。這個版本不估計自由 homography，而是只估相鄰
frame 的平移量，預設強制只沿 y 軸累積位移，避免 SIFT/homography 把地圖拉扯變形。

先用少量影片片段測試：

```bash
python3 scripts/stitch_vertical_strip.py \
  --input-dir raw_videos \
  --output stitched_outputs/vertical_strip_test.png \
  --frame-interval-seconds 2 \
  --duration-seconds 20 \
  --max-frames-per-video 10 \
  --overwrite
```

若要把 `raw_videos/video01.MP4`、`video02.MP4`、`video03.MP4` 依序連成一張圖，
但只丟掉第一支影片最前面幾秒、第三支影片最後面幾秒，可以使用：

```bash
python3 scripts/stitch_vertical_strip.py \
  --input-dir raw_videos \
  --output stitched_outputs/video01_02_03_vertical_strip.png \
  --skip-first-start-seconds 5 \
  --skip-last-end-seconds 5 \
  --frame-interval-seconds 2 \
  --max-frames-per-video 0 \
  --axis both \
  --render-scale 0.35 \
  --overwrite
```

如果已經先抽好 frame，也可以直接拼接影像資料夾：

```bash
python3 scripts/stitch_vertical_strip.py \
  --input-dir extracted_frames \
  --output stitched_outputs/vertical_strip_mosaic.png \
  --axis y \
  --smoothing median-step
```

若畫面有輕微左右漂移，可以允許 x/y 平移：

```bash
python3 scripts/stitch_vertical_strip.py \
  --input-dir extracted_frames \
  --output stitched_outputs/vertical_strip_xy_mosaic.png \
  --axis both
```

若速度確定很穩定，可以改成線性路徑平滑，讓整條航帶更接近等速模型：

```bash
python3 scripts/stitch_vertical_strip.py \
  --input-dir extracted_frames \
  --output stitched_outputs/vertical_strip_linear_mosaic.png \
  --smoothing linear-path
```

直接讀影片時，程式會把實際使用的影像存到 `<輸出檔名>_sampled_frames/`，
並同時輸出 `<輸出檔名>_report.json`。report 裡包含每張 frame 的累積位置、
相鄰 frame 的 phase correlation 位移與 response。若 response 很低，代表該段
重疊不足、畫面重複紋理太多，或跨影片銜接處不穩，建議先把該段 frame 分開處理。

### 手動微調航帶拼接結果

`scripts/adjust_strip_mosaic_gui.py` 可以讀取 `stitch_vertical_strip.py` 產生的
report，自動把飛行路徑切成多段直線與轉彎段，然後用 GUI 微調每一段的上下左右位置。

先檢查自動分段結果：

```bash
python3 scripts/adjust_strip_mosaic_gui.py \
  --report stitched_outputs/video01_02_03_vertical_strip_report.json \
  --print-segments
```

開啟 GUI：

```bash
python3 scripts/adjust_strip_mosaic_gui.py \
  --report stitched_outputs/video01_02_03_vertical_strip_report.json \
  --output stitched_outputs/video01_02_03_vertical_strip_adjusted.png \
  --render-scale 0.35 \
  --overwrite
```

如果要刪掉橫向轉彎段，只保留直線航帶，先輸出一張 vertical-only 基準圖：

```bash
python3 scripts/adjust_strip_mosaic_gui.py \
  --report stitched_outputs/video01_02_03_vertical_strip_report.json \
  --output stitched_outputs/video01_02_03_vertical_only.png \
  --adjustments stitched_outputs/video01_02_03_vertical_only_adjustments.json \
  --keep-label-prefix vertical \
  --render-scale 0.35 \
  --save-only \
  --overwrite
```

再開啟只含直線航帶的 GUI：

```bash
python3 scripts/adjust_strip_mosaic_gui.py \
  --report stitched_outputs/video01_02_03_vertical_strip_report.json \
  --output stitched_outputs/video01_02_03_vertical_only_adjusted.png \
  --adjustments stitched_outputs/video01_02_03_vertical_only_adjusted_adjustments.json \
  --keep-label-prefix vertical \
  --render-scale 0.35 \
  --overwrite
```

預設會開啟按鈕式 GUI，左側是控制面板，右側是較大的預覽畫面。
`Save Adjusted Mosaic` 與 `Quit` 固定在左上方，避免被其他控制項擠到看不到。
常用按鈕：

| 按鈕 | 功能 |
| --- | --- |
| `1. vertical...` 等段落按鈕 | 選取要調整的直線航帶 |
| `↑` / `↓` / `←` / `→` | 微調目前航帶 |
| `Previous` / `Next` | 切換上一段 / 下一段 |
| `Step -` / `Step +` | 縮小 / 放大每次移動距離 |
| `Zoom -` / `Zoom +` | 縮小 / 放大檢視 |
| `View ↑/↓/←/→` | 平移檢視畫面 |
| `Center` | 把目前航帶置中 |
| `Reset Segment` | 重設目前航帶 offset |
| `Reset All` | 重設所有航帶 offset |
| `Save Adjusted Mosaic` | 儲存調整後的大圖與 adjustment JSON |
| `Quit` | 離開 |

按鈕式 GUI 仍支援快捷鍵：

| 按鍵 | 功能 |
| --- | --- |
| 方向鍵或 `h/j/k/l` | 移動目前選取的航帶段 |
| `n` / `p` | 選下一段 / 上一段 |
| `+` / `-` | 放大 / 縮小每次微調步長 |
| `z` / `x` | 縮小 / 放大檢視 |
| `w/a/e/d` | 平移檢視畫面 |
| 滑鼠左鍵 | 選取點到的航帶段 |
| `c` | 把目前選取段置中 |
| `r` | 重設目前選取段的 offset |
| `0` | 重設所有段的 offset |
| `s` | 儲存調整後的大圖與 adjustment JSON |
| `q` 或 `Esc` | 離開 |

如果想使用舊版 OpenCV 快捷鍵視窗，可以加上：

```bash
python3 scripts/adjust_strip_mosaic_gui.py \
  --report stitched_outputs/video01_02_03_vertical_strip_report.json \
  --output stitched_outputs/video01_02_03_vertical_only_adjusted.png \
  --keep-label-prefix vertical \
  --render-scale 0.35 \
  --gui-backend opencv \
  --overwrite
```

### 框選長方形範圍另存新檔

`scripts/crop_image_roi.py` 可以從任意大圖中框選長方形 ROI，並用原始解析度裁切存成新檔。

互動框選：

```bash
python3 scripts/crop_image_roi.py \
  --input stitched_outputs/video01_02_03_vertical_only_adjusted.png \
  --output stitched_outputs/selected_roi.png \
  --overwrite
```

操作方式：

| 按鍵 / 操作 | 功能 |
| --- | --- |
| 滑鼠左鍵拖曳 | 框選長方形範圍 |
| `s` | 儲存目前框選範圍 |
| `r` | 重設框選 |
| `f` | 顯示整張圖 |
| `+` / `-` | 放大 / 縮小檢視 |
| `w/a/x/d` | 平移檢視畫面 |
| `q` 或 `Esc` | 離開不存檔 |

如果已經知道座標，也可以直接裁切：

```bash
python3 scripts/crop_image_roi.py \
  --input stitched_outputs/video01_02_03_vertical_only_adjusted.png \
  --output stitched_outputs/selected_roi.png \
  --x 1000 \
  --y 2000 \
  --width 1200 \
  --height 800 \
  --overwrite
```

程式會同時輸出 `<輸出檔名>_metadata.json`，記錄原圖大小與裁切座標。

## 地圖像素轉 GPS 與影像定位

目前預設以 `georeferenced_maps/localize_ready_selected_roi/uav_selected_roi_compressed_georef.json`
作為地理參考基準，對應的底圖是壓縮後的
`georeferenced_maps/localize_ready_selected_roi/uav_selected_roi_basemap_compressed.jpg`。
這張 UAV 底圖是 `4156 x 7925 px`，座標原點在左上角，x 往右增加，y 往下增加。
若要改用其他底圖，可以傳入 `--georef-json path/to/basemap_georef.json`。

舊的衛星圖 `衛星影像/aerial_gps_range_clean.png` 仍保留在 repo 中，可作為比較或備援參考。

### 建立自己的四點 GPS 底圖

如果要把自己選的圖片做成之後可重複使用的底圖，使用 `scripts/create_georeferenced_basemap.py`：

```bash
conda activate uav_contest_env
python3 scripts/create_georeferenced_basemap.py
```

如果不想切換目前 shell 的 conda 環境，也可以使用：

```bash
conda run --no-capture-output -n uav_contest_env python scripts/create_georeferenced_basemap.py
```

這個腳本需要在終端機輸入 GPS；不要用一般 `conda run -n ...`，否則點完 4 個點後可能會因為讀不到 stdin 而出現 `EOFError`。

程式會開啟檔案選擇視窗，讓你選圖片；接著在圖片上點 4 個已知 GPS 的控制點，按 `Enter` 或空白鍵確認後，在終端機依序輸入每個點的 `latitude,longitude`。這 4 個點不需要是長方形四個角，只要它們在圖片與 GPS 中的相對位置正確、且不要全部落在同一直線上，程式會用這些點估出整張圖共用的線性座標轉換。互動視窗支援：

| 按鍵 / 操作 | 功能 |
| --- | --- |
| 滑鼠左鍵 | 依序標記 `P1` 到 `P4` |
| `Enter` 或空白鍵 | 4 點都標好後繼續輸入 GPS |
| `u` | 復原上一個點 |
| `f` | 顯示整張圖 |
| `+` / `-` | 放大 / 縮小檢視 |
| `w/a/s/d` | 平移檢視畫面 |
| `q` 或 `Esc` | 離開不存檔 |

預設會輸出到 `georeferenced_maps/<圖片檔名>/`：

```text
georeferenced_maps/<圖片檔名>/
├── basemap.<原副檔名>
├── basemap_georef.json
├── basemap_georef_preview.jpg
└── control_points.csv
```

`basemap_georef.json` 會保存圖片尺寸、4 個控制點、WGS84 GPS、pixel 到 GPS 的 affine 線性轉換，以及 GPS 回 pixel 的反向 affine 線性轉換。後續轉換座標時，把這份 JSON 傳給 `--georef-json`：

```bash
python3 scripts/georeference_map.py pixel \
  --georef-json georeferenced_maps/my_map/basemap_georef.json \
  --x 801 \
  --y 258
```

反向從 GPS 換回底圖 pixel：

```bash
python3 scripts/georeference_map.py gps \
  --georef-json georeferenced_maps/my_map/basemap_georef.json \
  --lat 23.45564 \
  --lon 120.28169
```

也可以直接用這份底圖 JSON 跑影像匹配或車輛定位：

```bash
python3 scripts/georeference_map.py match \
  --georef-json georeferenced_maps/my_map/basemap_georef.json \
  --query path/to/query_image.jpg \
  --output stitched_outputs/georef/query_match.png

python3 scripts/localize_vehicles.py \
  --georef-json georeferenced_maps/my_map/basemap_georef.json \
  --frame test_image/frame_000051.jpg
```

產生一張帶有格線、邊界與已知 GPS 點位的基準圖：

```bash
python3 scripts/georeference_map.py draw-map
```

若要產生後直接開視窗查看，加上 `--show`：

```bash
python3 scripts/georeference_map.py draw-map --show
```

預設輸出：

```text
stitched_outputs/georef/aerial_reference_grid.png
```

如果要手動點擊地圖並輸出 GPS：

```bash
python3 scripts/georeference_map.py click
```

程式會先輸出同一張基準圖，然後開啟 OpenCV 視窗。對地圖左鍵點擊時，終端機會輸出該 pixel 對應的 WGS84 GPS，例如：

```json
{
  "x": 801.0,
  "y": 258.0,
  "latitude": 23.45563935,
  "longitude": 120.2816845
}
```

如果只想從命令列換算單一 pixel：

```bash
python3 scripts/georeference_map.py pixel --x 801 --y 258
```

反向從 GPS 換算回地圖 pixel：

```bash
python3 scripts/georeference_map.py gps --lat 23.45564 --lon 120.28169
```

如果要輸入一張影像，讓程式在基準地圖上找最符合的位置：

```bash
python3 scripts/georeference_map.py match \
  --query path/to/query_image.jpg \
  --output stitched_outputs/georef/query_match.png \
  --show
```

`match` 模式會先嘗試 SIFT/ORB 特徵匹配與 RANSAC homography；若特徵不足，`auto` 模式會退回多尺度 template matching。輸出 JSON 會包含匹配中心點的 pixel 與 GPS，並輸出一張視覺化圖片，把 query 的估計位置畫回目前使用的 georeferenced basemap。加上 `--show` 時，程式會在存檔後直接開 OpenCV 視窗顯示結果，按 `q` 或 `Esc` 關閉。如果 query 影像和基準地圖比例、角度或透視差異很大，匹配信心會下降；正式流程仍建議先用無人機 GPS/IMU 粗估候選區域，再在局部範圍內做影像配準。

如果輸入影像的上下左右方向不固定，可以加 `--orientations` 讓程式嘗試多個方向後選分數最高者：

```bash
python3 scripts/georeference_map.py match \
  --query path/to/query_image.jpg \
  --orientations all \
  --output stitched_outputs/georef/query_match.png \
  --show
```

可選值：

- `none`: 只用原圖，預設值。
- `rotations`: 嘗試原圖與 90/180/270 度旋轉。
- `flips`: 嘗試原圖、水平翻轉、垂直翻轉與 180 度旋轉。
- `all`: 嘗試旋轉、翻轉與對角轉置共 8 種方向。

對 `extracted_frames/` 這類 `3840 x 2160` 無人機 frame，整張圖直接全域匹配通常不可靠，因為近距離農田紋理會和衛星圖中很多位置相似。建議用 `--query-roi x,y,width,height` 只截取道路、河道、建物邊緣等固定地物；如果要把原始 frame 內某個點一起換成 GPS，可加 `--query-point x,y`：

```bash
python3 scripts/georeference_map.py match \
  --query extracted_frames/frame_000073.jpg \
  --query-roi 0,900,1600,900 \
  --query-point 800,1350 \
  --orientations all \
  --output stitched_outputs/georef/frame_000073_roi_match.png \
  --show
```

如果輸出含有 `Low template score` warning，該結果只能當粗略猜測，不建議當成最終 GPS。這時應該縮小 ROI、改選更穩定的固定地物，或先用無人機 GPS/IMU 限定候選區域。

## 範例：自動找車並標記地圖座標

比賽規章要求飛行展示過程中要在回傳畫面框出辨識成功目標與座標；賽後繳交紙本結果時，座標需以 TWD97 呈現。`scripts/localize_vehicles.py` 針對這個需求做一個離線 demo pipeline：

1. 讀取無人機 frame，例如 `test_image/frame_000051.jpg`。
2. 用 tile + upscale 方式把小車放大後丟入車輛偵測器。
3. 把整張 frame 對到預設壓縮 UAV georeferenced basemap。
4. 將車輛中心點轉成參考大地圖 pixel、WGS84 GPS 與 TWD97 TM2 座標。
5. 輸出原圖框選、地圖標記、流程總覽圖、JSON 與 CSV。

安裝 YOLO 相關套件後，預設會使用 YOLO26x 權重，先找 `car` 類別，再用 target verifier 檢查白車上是否有紅色或粉紅色圖案：

```bash
python3 -m pip install -r requirements.txt

python3 scripts/localize_vehicles.py \
  --frame test_image/frame_000051.jpg \
  --detector yolo \
  --yolo-model yolo26x.pt \
  --vehicle-classes car \
  --imgsz 1280 \
  --tile-size 960 \
  --tile-overlap 240 \
  --tile-upscales 1,2 \
  --target-verifier \
  --show
```

如果本機尚未安裝 `ultralytics` 或暫時沒有 YOLO26x 權重，可以先用白車 heuristic 產生 demo 圖，確認輸出格式與視覺化流程：

```bash
python3 scripts/localize_vehicles.py \
  --frame test_image/frame_000051.jpg \
  --detector white-heuristic \
  --show
```

主要輸出：

```text
vehicle_localization_outputs/frame_000051/
├── 01_frame_vehicle_detections.jpg
├── 02_map_vehicle_coordinates.jpg
├── 03_process_overview.jpg
├── vehicle_localization.json
├── vehicle_localization.csv
└── crops/
```

如果沒有指定 `--output-dir`，程式會依輸入檔名自動建立獨立輸出資料夾，例如 `--frame test_image/frame_000161.jpg` 會輸出到：

```text
vehicle_localization_outputs/frame_000161/
```

若要用本機 `uav_contest_env` 重新產生 GitHub 內附的兩組測試輸出，執行：

```bash
conda run -n uav_contest_env python scripts/localize_vehicles.py \
  --frame test_image/frame_000051.jpg \
  --detector yolo \
  --yolo-model yolo26x.pt \
  --vehicle-classes car \
  --imgsz 1280 \
  --tile-upscales 1,2

conda run -n uav_contest_env python scripts/localize_vehicles.py \
  --frame test_image/frame_000161.jpg \
  --detector yolo \
  --yolo-model yolo26x.pt \
  --vehicle-classes car \
  --imgsz 1280 \
  --tile-upscales 1,2
```

注意：`frame_000051.jpg` 這類全圖對衛星圖做特徵匹配時容易被重複農田紋理誤導，因此 script 預設用 template matching，並會在 JSON 的 `warnings` 記錄低信心匹配。正式比賽版本應加入無人機 GPS/IMU 或穩定固定地物 ROI 來縮小搜尋範圍。地圖匹配預設會測試 `0/90/180/270` 度旋轉，並在車輛中心點轉地圖座標時套用對應旋轉矩陣。為了避免重複田地紋理造成弱假匹配，預設只有旋轉方向分數比原方向高出 `--orientation-switch-margin 0.08` 以上才會切換；如果確定影像方向固定，可加 `--orientations none` 只測原始方向以加速。

如果車輛中心點投影後超出目前 georeferenced map 範圍，程式會改用最靠近的地圖邊界點產生 GPS/TWD97，確保輸出仍有數值。JSON 會標記 `map_pixel_clamped: true`、保留 `original_map_pixel`，並在 `warnings` 寫出使用邊界點的原因。

加上 `--show` 時，程式會在輸出檔案後開啟 `03_process_overview.jpg` 的 OpenCV 視窗。這張 overview 會用較高解析度重新排版左右圖和座標表；下方會列出每台車的 image center、WGS84 與 TWD97 座標，按 `q` 或 `Esc` 關閉。

目前初賽目標 verifier 的策略是「先找車，再在車框內確認白車與紅色/粉紅色圖案」；白色叉叉形狀只作為輔助視覺線索，不作為硬性門檻。

## 使用 UDIS++ 拼接影像

`scripts/stitch_frames_udis2.py` 是 UDIS++ 的 adapter，不直接包含 UDIS2 原始碼或模型權重。它會把 `extracted_frames/` 內剩下的 frame 依檔名排序，逐對送進官方 UDIS2 的 Stage 1 Warp 和 Stage 2 Composition，並把每一輪的 `composition.jpg` 當成下一輪輸入，最後輸出一張 progressive mosaic。

這個版本比較接近 UDIS++ 論文方法，但需要額外環境與 checkpoint，速度也會比 OpenCV 版本慢很多。它適合用來追求 seam/fusion 品質，不適合快速大量試參數。

先 clone 官方 UDIS2 repo：

```bash
mkdir -p third_party
git clone https://github.com/nie-lang/UDIS2.git third_party/UDIS2
```

依 UDIS2 官方 README 下載兩個 pretrained model，分別放到：

```text
third_party/UDIS2/Warp/model/
third_party/UDIS2/Composition/model/
```

建議另外建立 UDIS2 專用 Python/conda 環境。本 repo 目前測試過 Python 3.13 搭配 PyTorch cu132：

```bash
conda env create -f environment-udis2-py313.yml
conda activate udis2_py313

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip3 install opencv-python scikit-image gdown
```

如果你想保留 UDIS2 官方較舊的 Python 3.8 / PyTorch 1.7 路線，也可以參考 `environment-udis2.yml`；但目前本機成功 smoke test 的環境是 `udis2_py313`。

本機已建立的 Python 3.13 環境路徑是：

```text
/home/steve/anaconda3/envs/udis2_py313/bin/python
```

下載官方 checkpoint：

```bash
conda run -n udis2_py313 gdown --id 1GBwB0y3tUUsOYHErSqxDxoC_Om3BJUEt \
  -O third_party/UDIS2/Warp/model/warp.pth

conda run -n udis2_py313 gdown --id 1OaG0ayEwRPhKVV_OwQwvwHDFHC26iv30 \
  -O third_party/UDIS2/Composition/model/composition.pth
```

執行 UDIS++ progressive 拼接：

```bash
python3 scripts/stitch_frames_udis2.py \
  --input-dir extracted_frames \
  --udis2-root third_party/UDIS2 \
  --output-dir stitched_outputs/udis2 \
  --udis2-python /home/steve/anaconda3/envs/udis2_py313/bin/python \
  --gpu -1
```

如果你在 conda 環境中安裝 UDIS2，可以指定那個環境的 Python：

```bash
python3 scripts/stitch_frames_udis2.py \
  --udis2-python /path/to/conda/envs/udis2_py313/bin/python
```

常用參數：

- `--gpu -1`: 使用 CPU，適合 NVIDIA driver 或 CUDA 還沒確認正常時。
- `--gpu 0`: 傳給 UDIS2 的 GPU id；只有在 `nvidia-smi` 和 PyTorch CUDA tensor 測試都正常時再使用。
- `--max-iter 50`: 每一對影像做 warp adaption 的迭代次數；越高通常越慢。
- `--overwrite`: 覆蓋既有輸出資料夾。
- `--clean-work`: 輸出最後大圖後刪除每一對影像的中間資料夾；預設會保留中間輸出，方便檢查 `warp1.jpg`、`warp2.jpg`、`mask1.jpg`、`mask2.jpg`、`composition.jpg`。

CUDA 檢查方式：

```bash
nvidia-smi
conda run -n udis2_py313 python -c "import torch; print(torch.cuda.is_available()); print(torch.zeros(1).cuda())"
```

如果上述任一指令失敗或卡住，先用 `--gpu -1`。

預設最後輸出：

```text
stitched_outputs/udis2/udis2_mosaic.jpg
```

注意：UDIS2 官方 `test_other.py` 是針對兩張影像設計；本專案腳本採用逐對 progressive 的方式擴展到多張 frame。若任務路徑很長，可能會累積誤差，必要時可以先刪掉模糊、重疊太少或頭尾不需要的 frame。
