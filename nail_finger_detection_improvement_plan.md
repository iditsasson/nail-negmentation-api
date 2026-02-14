# Nail & Finger Detection Improvement Plan

## Goal
Replace the current polygon-geometry-based orientation detection with proper hand keypoints estimation and pose detection using YOLO. This will give accurate finger direction, hand pose, and per-finger identification instead of heuristic ellipse fitting.

## Current Approach (v1)
- `compute_nail_orientation()` in `segmentation_engine.py`
- Uses `cv2.fitEllipse()` on nail segmentation polygons
- Determines tip vs cuticle by comparing perpendicular spread of polygon halves
- Limitations: no finger identity, no hand pose context, heuristic-only

## Planned Approach (v2)
Use Ultralytics YOLO pose estimation trained on hand keypoints to detect 21 hand landmarks per hand. Combine with existing nail segmentation to:
1. Identify which finger each detected nail belongs to
2. Get accurate finger direction from keypoint pairs (e.g., MCP → fingertip vector)
3. Detect hand pose (open, fist, pointing, etc.)

## Resources

### Hand Keypoints Dataset
- **URL:** https://docs.ultralytics.com/datasets/pose/hand-keypoints/
- 26,782 images (train: 21,170, val: 5,612) with 21 keypoints per hand
- COCO-pose format, compatible with Ultralytics training pipeline
- 21 keypoints: wrist + 4 joints per finger (thumb, index, middle, ring, pinky)

### Ultralytics Blog Guide (YOLO11)
- **URL:** https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11
- Step-by-step guide for training YOLO11-pose on the hand keypoints dataset
- Covers dataset setup, training config, and inference
- Can be adapted for YOLO26 when available

### Community Implementation (YOLO11n-pose-hands)
- **URL:** https://github.com/chrismuntean/YOLO11n-pose-hands/tree/main/runs/pose/train
- Pre-trained YOLO11n-pose model specifically for hand keypoints
- Contains training runs/results to reference for expected performance
- Worth checking their training config and metrics

## Architecture Notes

### Integration Strategy
- Run hand keypoints model alongside (or instead of) current nail segmentation
- Match detected nails to nearest finger keypoints to assign finger identity
- Compute orientation from keypoint vectors instead of polygon geometry
- The 21 keypoints per hand follow this layout:
  - 0: Wrist
  - 1-4: Thumb (CMC, MCP, IP, tip)
  - 5-8: Index finger (MCP, PIP, DIP, tip)
  - 9-12: Middle finger (MCP, PIP, DIP, tip)
  - 13-16: Ring finger (MCP, PIP, DIP, tip)
  - 17-20: Pinky (MCP, PIP, DIP, tip)

### Finger Direction from Keypoints
For each finger, direction = vector from MCP joint to fingertip:
- Thumb: keypoint 2 → keypoint 4
- Index: keypoint 5 → keypoint 8
- Middle: keypoint 9 → keypoint 12
- Ring: keypoint 13 → keypoint 16
- Pinky: keypoint 17 → keypoint 20

### Model Options
- **YOLO26-pose** (preferred, when released) — latest architecture
- **YOLO11n-pose** (available now) — lightweight, proven on hand keypoints
- Consider nano (n) variant for CPU deployment, small (s) for GPU
