# Localize-ready selected ROI basemap

Use the compressed map for normal runs:

```bash
python scripts/localize_vehicles.py \
  --georef-json georeferenced_maps/localize_ready_selected_roi/uav_selected_roi_compressed_georef.json \
  --frame test_image/frame_000051.jpg
```

Use the original PNG map if you need lossless image quality:

```bash
python scripts/localize_vehicles.py \
  --georef-json georeferenced_maps/localize_ready_selected_roi/uav_selected_roi_georef.json \
  --frame test_image/frame_000051.jpg
```

Image size: 4156 x 7925 pixels.
Compressed image keeps the same pixel dimensions, so GPS conversion results are unchanged.
