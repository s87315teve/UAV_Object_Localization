# Video03 Target Test Video

This test clip is a degraded version of `raw_videos/video03.MP4` for target-verifier and localization smoke tests.

Output files:

- `augmented_test_data/videos/video03_target_low_quality.mp4`: low-quality MP4, under 30 MB.
- `augmented_test_data/videos/video03_target_low_quality.metadata.json`: generation settings and per-output-frame target bounding boxes.
- `augmented_test_data/videos/video03_target_low_quality.preview.jpg`: first generated frame for quick visual inspection.

The generator downsamples video03, lowers frame rate, adds blur/noise/JPEG artifacts, and encodes with a low H.264 bitrate. It pastes one target vehicle into every output frame. The target vehicle uses the same red marker with a thick white X style as `scripts/generate_target_verifier_test_data.py`.

By default the pasted vehicle is the real white-car crop at `vehicle_localization_outputs/frame_000051/crops/veh_001.jpg`; the script preserves the crop RGB content and only removes the crop background before adding the target marker.

Regenerate:

```bash
python scripts/generate_video03_target_test_video.py --overwrite
```

The script verifies that the output MP4 is below `--max-size-mb` before writing metadata. Use `--bitrate`, `--fps`, or `--output-width` to make the file smaller if needed.
