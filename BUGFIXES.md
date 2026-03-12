# Bug Fix Log

All timestamps are local time (America/Los_Angeles).

## 2026-03-11 22:23:57 PDT
- Fixed missing track dimension attributes by defining `track_width_px` and `track_height_px` on `Track` and updating the renderer to use them. Files: `rl_car_rl/env/track.py`, `rl_car_rl/visualization/renderer.py`.
- Fixed device mismatch in policy action sampling (MPS vs CPU) by ensuring distribution tensors and sampled actions are on the same device. File: `rl_car_rl/agent/policy.py`.
- Smoothed track boundary generation to reduce corner artifacts by averaging tangents and increasing spline resolution. File: `rl_car_rl/env/track.py`.
