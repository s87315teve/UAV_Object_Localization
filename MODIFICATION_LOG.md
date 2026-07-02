# 修改紀錄

## 2026-07-02

- 分支：`vehicle-localization-test-assets`，預計合併至 `main`
- 變更摘要：將車輛定位 demo 的預設輸出改為獨立資料夾 `vehicle_localization_outputs/`，並加入 frame 51 與 frame 161 的輸出結果，方便直接從 GitHub 取得測試輸入與對應輸出。
- 影響範圍：`scripts/localize_vehicles.py`、`README.md`、`.gitignore`、`vehicle_localization_outputs/`、`MODIFICATION_LOG.md`

## 2026-07-02

- 分支：`vehicle-localization-test-assets`
- 變更摘要：新增 UAV frame 車輛偵測與地圖座標標記流程，支援 YOLO26m tile/upscale 偵測、白車 fallback、四方向地圖匹配、旋轉後車輛中心點定位、WGS84/TWD97 輸出、demo overview 視覺化與測試影像資料夾。
- 影響範圍：`scripts/localize_vehicles.py`、`scripts/georeference_map.py`、`README.md`、`requirements.txt`、`.gitignore`、`test_image/`、`MODIFICATION_LOG.md`

## 2026-07-01

- 分支：`main`
- 變更摘要：新增地理參考地圖工具，可把 `aerial_gps_range_clean.png` 的 pixel 座標轉成 GPS、產生可視化基準圖、互動點擊輸出 GPS，並用輸入影像匹配回基準地圖位置；影像匹配支援大圖自動縮放、低分警告、`--query-roi`、`--query-point`、`--orientations` 方向候選與 `--show` 視窗顯示。
- 影響範圍：`scripts/georeference_map.py`、`README.md`、`MODIFICATION_LOG.md`

## 2026-06-30

- 分支：`main`
- 變更摘要：新增 Python 3.13 / PyTorch cu132 的 UDIS2 環境說明與環境檔，並加入 CPU compatibility hook 讓新版 PyTorch 可執行 UDIS2 smoke test。
- 影響範圍：`environment-udis2-py313.yml`、`scripts/stitch_frames_udis2.py`、`scripts/udis2_compat/sitecustomize.py`、`README.md`、`MODIFICATION_LOG.md`

## 2026-06-30

- 分支：`main`
- 變更摘要：新增 UDIS2 conda 環境檔與 checkpoint 下載教學，並將 UDIS++ adapter 預設改為 CPU 模式避免 CUDA driver 異常時卡住。
- 影響範圍：`environment-udis2.yml`、`scripts/stitch_frames_udis2.py`、`README.md`、`MODIFICATION_LOG.md`

## 2026-06-30

- 分支：`main`
- 變更摘要：新增 UDIS++ progressive 拼接 adapter，並在 README 補充 UDIS2 repo、checkpoint、執行方式與限制說明。
- 影響範圍：`scripts/stitch_frames_udis2.py`、`README.md`、`.gitignore`、`MODIFICATION_LOG.md`

## 2026-06-30

- 分支：`main`
- 變更摘要：新增抽取影像拼接成大圖的腳本，支援黑色空白區、feather blending、CUDA flag 與 README 使用教學。
- 影響範圍：`scripts/stitch_frames.py`、`README.md`、`.gitignore`、`requirements.txt`、`MODIFICATION_LOG.md`

## 2026-06-30

- 分支：`main`
- 變更摘要：新增影片每 3 秒抽取影像的腳本，並在 README 補充執行方式與輸出說明。
- 影響範圍：`scripts/extract_frames.py`、`README.md`、`.gitignore`、`MODIFICATION_LOG.md`

## 2026-06-30

- 分支：`main`
- 變更摘要：新增 repository agent 工作規則，包含遠端版本檢查、修改紀錄更新與既有變更保護流程。
- 影響範圍：`AGENTS.MD`、`MODIFICATION_LOG.md`

## 2026-06-30

- 分支：`main`
- 變更摘要：新增 README 專案說明、流程摘要、MVP 建議、輸出格式與評估重點；在 README 最上方加入修改紀錄連結。
- 影響範圍：`README.md`、`MODIFICATION_LOG.md`
