[修改紀錄](MODIFICATION_LOG.md)

# UAV Object Localization

本專案目標是建立一套無人機地面車輛定位流程：使用無人機拍攝目標區域影像，辨識影像中的車輛，並把車輛在影像中的位置轉換成地圖上的 GPS 座標。

完整流程說明請見 [docs/drone_vehicle_geolocation_workflow.md](docs/drone_vehicle_geolocation_workflow.md)。

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

## 地圖像素轉 GPS 與影像定位

目前先以 `衛星影像/aerial_gps_range_clean.png` 作為地理參考基準圖，GPS 範圍與圖片尺寸記錄在 `衛星影像/aerial_gps_range_info.md`。這張圖是 `1600 x 1000 px`，座標原點在左上角，x 往右增加，y 往下增加。換算只保證在該檔案記錄的 GPS 外框內有效，超出外框的點會被忽略或直接報錯。

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

`match` 模式會先嘗試 SIFT/ORB 特徵匹配與 RANSAC homography；若特徵不足，`auto` 模式會退回多尺度 template matching。輸出 JSON 會包含匹配中心點的 pixel 與 GPS，並輸出一張視覺化圖片，把 query 的估計位置畫回 `aerial_gps_range_clean.png`。加上 `--show` 時，程式會在存檔後直接開 OpenCV 視窗顯示結果，按 `q` 或 `Esc` 關閉。如果 query 影像和基準地圖比例、角度或透視差異很大，匹配信心會下降；正式流程仍建議先用無人機 GPS/IMU 粗估候選區域，再在局部範圍內做影像配準。

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
3. 把整張 frame 對到 `衛星影像/aerial_gps_range_clean.png`。
4. 將車輛中心點轉成參考大地圖 pixel、WGS84 GPS 與 TWD97 TM2 座標。
5. 輸出原圖框選、地圖標記、流程總覽圖、JSON 與 CSV。

安裝 YOLO 相關套件後，預設會使用 YOLO26l 權重：

```bash
python3 -m pip install -r requirements.txt

python3 scripts/localize_vehicles.py \
  --frame test_image/frame_000051.jpg \
  --detector yolo \
  --yolo-model yolo26l.pt \
  --tile-size 960 \
  --tile-overlap 240 \
  --tile-upscales 1,2 \
  --show
```

如果本機尚未安裝 `ultralytics` 或暫時沒有 YOLO26l 權重，可以先用白車 heuristic 產生 demo 圖，確認輸出格式與視覺化流程：

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
  --yolo-model yolo26l.pt

conda run -n uav_contest_env python scripts/localize_vehicles.py \
  --frame test_image/frame_000161.jpg \
  --detector yolo \
  --yolo-model yolo26l.pt
```

注意：`frame_000051.jpg` 這類全圖對衛星圖做特徵匹配時容易被重複農田紋理誤導，因此 script 預設用 template matching，並會在 JSON 的 `warnings` 記錄低信心匹配。正式比賽版本應加入無人機 GPS/IMU 或穩定固定地物 ROI 來縮小搜尋範圍。地圖匹配預設會測試 `0/90/180/270` 度旋轉，並在車輛中心點轉地圖座標時套用對應旋轉矩陣。為了避免重複田地紋理造成弱假匹配，預設只有旋轉方向分數比原方向高出 `--orientation-switch-margin 0.08` 以上才會切換；如果確定影像方向固定，可加 `--orientations none` 只測原始方向以加速。

加上 `--show` 時，程式會在輸出檔案後開啟 `03_process_overview.jpg` 的 OpenCV 視窗。這張 overview 下方會列出每台車的 image center、WGS84 與 TWD97 座標，按 `q` 或 `Esc` 關閉。

目前先做「車輛」辨識，車頂 80cm x 80cm 指認圖或白車上的 X 圖案辨識先列為後續工作；建議下一步收集比賽高度下的車頂圖案樣本，另外訓練 marker detector 或在 YOLO 車框內做二階段分類。

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
