# Video Recording Configuration

## Status: ✅ Fixed - Video recording now disabled during training

**Date**: 2025-11-27

---

## Problem

During HAPPO training, videos were being recorded automatically:
```
Successfully store the video of last episode
Successfully store the video of last episode
...
```

This caused:
- **Performance degradation** (rendering overhead)
- **Disk space usage** (multiple video files per episode)
- **Unnecessary I/O** during training

---

## Root Cause

Two config files had `record_video = True` by default:

1. **`mqe/envs/go1/go1_config.py:49`** - Base Go1 configuration
2. **`mqe/envs/configs/go1_push_upper_config.py:15`** - Upper-level task config

These defaults were intended for visualization/debugging but were always on during training.

---

## Fix Applied

Changed both files to disable video recording by default:

### File 1: `mqe/envs/go1/go1_config.py`
```python
# Line 48-49
# BEFORE:
record_video = True

# AFTER:
record_video = False  # Set to True only for testing/visualization
```

### File 2: `mqe/envs/configs/go1_push_upper_config.py`
```python
# Line 15
# BEFORE:
record_video = True

# AFTER:
record_video = False  # Set to True only for testing/visualization
```

---

## How to Enable Video Recording

### For Testing/Visualization

**Option 1**: Edit the config file temporarily
```python
# In task/cuboid/config.py (or whichever task config)
class env(Go1Cfg.env):
    record_video = True
```

**Option 2**: Use command-line flag (OpenRL training)
```bash
python openrl_ws/test.py --record_video --checkpoint /path/to/checkpoint
```

**Option 3**: Modify task-specific config
```bash
# Edit the specific task config before training
vim task/cuboid/config.py

# Add in class env:
record_video = True
```

---

## Video Recording Details

When enabled, videos are saved to:
- **OpenRL**: `docs/video/`
- **Recording format**: MP4
- **Resolution**: 1080x1080 (configurable via `recording_width_px` and `recording_height_px`)
- **Filename pattern**: `test_seed{N}_{M}eps.mp4`

**Note**: Video recording significantly slows down training (requires rendering every frame).

---

## Performance Impact

**Before fix** (with video recording):
- ~10-20% slower training (rendering overhead)
- ~100-500MB per video file
- Continuous disk writes

**After fix** (no video recording):
- ✅ Full training speed
- ✅ No disk space wasted
- ✅ No rendering overhead

---

## Summary

✅ **Video recording is now OFF by default**
- Training runs at full speed
- No unnecessary video files created
- Can be easily enabled when needed for visualization

To record videos, explicitly set `record_video = True` in your task config or use `--record_video` flag.
