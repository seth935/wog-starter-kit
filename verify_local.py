"""Weights of Giants - Local Pre-Submission Verification Script (wog-cv-v2.1).

Run this script inside your repository before submitting to the arena:
    python verify_local.py
    python verify_local.py --image path/to/test.jpg --save-preview preview.jpg

Checks:
1. Manifest validity (`wog.yaml` / `model.json`) against the `la-frite-1gb` spec
   (`onnxruntime`, `input_size <= 512`, `threads <= 4`, `frame_policy.skip == 0`,
   `conf_threshold`, `iou_threshold`, `bytetrack-lite`).
2. Unsafe deserialization artifacts (`.pkl`, `.pickle`, `.pt`, `.pth`, `.joblib`, `.bin`).
3. Weight file size limits (`<= 40 MB`) and ONNX IR/opset ceiling (`IR <= 10`, `opset <= 22`).
4. Exact La Frite `dut_shim.py` preprocessing (`letterbox` pad=114 / `stretch` / `crop`, RGB `[0,1]`),
   output shape (`[1, N, 6|7|8]`), and numeric range validation (no NaN/Inf, normalized `[0,1]` boxes,
   `theta` in radians, `conf` in `[0,1]`, `cls` in `{0, 1}`).
5. Optional `--image` + `--save-preview` visual check on a real image.
6. Note: Peak RAM (`<= 800 MB`, full credit `<= 205 MB`) and official FPS (`>= 2.4 FPS`,
   full credit `>= 10.0 FPS`) are measured only on the physical La Frite Cortex-A53 board.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import onnxruntime as ort
except ImportError:
    print("Error: numpy and onnxruntime are required to run verify_local.py.")
    print("Install them with: pip install numpy onnxruntime")
    sys.exit(1)

CORTEX_A53_PROFILE = {
    "limits": {
        "artifact_mb": 40,
        "peak_rss_mb": 800,
        "memory_full_credit_mb": 205,
        "fps_floor": 2.4,
        "fps_full_marks": 10.0,
        "max_threads": 4,
        "max_input_side_px": 512,
        "max_onnx_ir": 10,
        "max_onnx_opset": 22,
    }
}
ALLOWED_RESIZE_MODES = {"letterbox", "stretch", "crop"}
ALLOWED_DOMAINS = {"indoor", "urban", "rural"}
SKIP_SCAN_DIRS = {".git", ".venv", "venv", "__pycache__"}
UNSAFE_EXTS = {".pkl", ".pickle", ".pt", ".pth", ".joblib", ".bin"}


def preprocess_like_dut(frame_rgb: np.ndarray, wh: tuple[int, int], resize_mode: str = "letterbox"):
    """Exact mirror of backend/rig/dut_shim.py ModelRunner._preprocess."""
    w, h = int(wh[0]), int(wh[1])
    orig_h, orig_w = frame_rgb.shape[:2]
    transform = (0.0, 0.0, float(w), float(h))
    if orig_h != h or orig_w != w:
        import cv2

        if resize_mode == "letterbox":
            r = min(w / float(orig_w), h / float(orig_h))
            new_w = max(1, int(round(orig_w * r)))
            new_h = max(1, int(round(orig_h * r)))
            resized = cv2.resize(frame_rgb, (new_w, new_h))
            canvas = np.full((h, w, 3), 114, dtype=np.uint8)
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            frame_rgb = canvas
            transform = (float(pad_x), float(pad_y), float(new_w), float(new_h))
        elif resize_mode == "crop":
            r = max(w / float(orig_w), h / float(orig_h))
            scaled_w = max(w, int(round(orig_w * r)))
            scaled_h = max(h, int(round(orig_h * r)))
            resized = cv2.resize(frame_rgb, (scaled_w, scaled_h))
            crop_x = (scaled_w - w) // 2
            crop_y = (scaled_h - h) // 2
            frame_rgb = resized[crop_y : crop_y + h, crop_x : crop_x + w]
            transform = (-float(crop_x), -float(crop_y), float(scaled_w), float(scaled_h))
        else:
            frame_rgb = cv2.resize(frame_rgb, (w, h))
    x = (frame_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(x), transform


def _validate_output_values(detections: np.ndarray) -> tuple[list[str], list[str]]:
    """Validate tensor shape and numeric contract matching backend/rig/dut_shim.py."""
    failures: list[str] = []
    warnings: list[str] = []

    a = np.asarray(detections)
    if a.ndim < 2 or a.shape[-1] not in (6, 7, 8):
        failures.append(
            f"Invalid output shape: {a.shape}. Expected [1, N, 7] (OBB), [1, N, 8] (OBB+id), or [1, N, 6] (BB). "
            "Wrap raw YOLO heads with export_onnx.py."
        )
        return failures, warnings

    if not np.all(np.isfinite(a)):
        failures.append("Model output contains NaN or Inf values.")
        return failures, warnings

    rows = a.reshape(-1, a.shape[-1])
    cols = rows.shape[-1]
    conf_col = 4 if cols == 6 else 5
    cls_col = 5 if cols == 6 else 6

    confs = rows[:, conf_col]
    if np.any(confs < -1e-3) or np.any(confs > 1.001):
        failures.append(
            f"Confidence column out of [0, 1] range (min={float(confs.min()):.3f}, max={float(confs.max()):.3f}). "
            "Apply sigmoid/softmax inside the ONNX graph."
        )

    active = rows[confs >= 0.10]
    if len(active) > 0:
        coords = active[:, :4]
        if float(coords.max()) > 1.5 or float(coords.min()) < -0.5:
            failures.append(
                f"Active box coordinates look unnormalized (min={float(coords.min()):.2f}, max={float(coords.max()):.2f}). "
                "Output [cx, cy, w, h] normalized to [0, 1] relative to [input_w, input_h]."
            )
        if cols >= 7:
            thetas = active[:, 4]
            if float(np.max(np.abs(thetas))) > math.pi + 0.05:
                failures.append(
                    f"Angle theta exceeds radians range (max |theta|={float(np.max(np.abs(thetas))):.2f}). "
                    "Output theta in radians (canonical [-pi/2, pi/2)), not degrees."
                )
        classes = np.rint(active[:, cls_col]).astype(int)
        bad_cls = sorted(set(int(c) for c in classes if c not in (0, 1)))
        if bad_cls:
            failures.append(
                f"Unexpected class IDs {bad_cls} in active detections. "
                "Season Zero requires 0=person, 1=vehicle (use export_onnx.py --coco-remap for COCO models)."
            )
    else:
        warnings.append(
            "Model produced 0 detections >= 0.10 on the random smoke tensor "
            "(normal for real detectors on noise; test with --image to verify real boxes)."
        )

    return failures, warnings


def _obb_iou_nms(a: dict, b: dict) -> float:
    import cv2

    w1, h1, th1 = a["w"], a["h"], a["theta"]
    w2, h2, th2 = b["w"], b["h"], b["theta"]
    if w1 <= 1e-7 or h1 <= 1e-7 or w2 <= 1e-7 or h2 <= 1e-7:
        return 0.0
    if abs(th1) < 1e-4 and abs(th2) < 1e-4:
        ix1 = max(a["cx"] - w1 * 0.5, b["cx"] - w2 * 0.5)
        iy1 = max(a["cy"] - h1 * 0.5, b["cy"] - h2 * 0.5)
        ix2 = min(a["cx"] + w1 * 0.5, b["cx"] + w2 * 0.5)
        iy2 = min(a["cy"] + h1 * 0.5, b["cy"] + h2 * 0.5)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = w1 * h1 + w2 * h2 - inter
        return float(inter / union) if union > 1e-9 else 0.0
    r1 = ((float(a["cx"]), float(a["cy"])), (float(w1), float(h1)), float(math.degrees(th1)))
    r2 = ((float(b["cx"]), float(b["cy"])), (float(w2), float(h2)), float(math.degrees(th2)))
    kind, pts = cv2.rotatedRectangleIntersection(r1, r2)
    if kind == cv2.INTERSECT_NONE or pts is None or len(pts) < 3:
        return 0.0
    inter = float(cv2.contourArea(cv2.convexHull(pts.reshape(-1, 2).astype(np.float32))))
    union = w1 * h1 + w2 * h2 - inter
    return float(max(0.0, min(1.0, inter / union))) if union > 1e-9 else 0.0


def _decode_and_draw_preview(
    image_path: str,
    session: ort.InferenceSession,
    input_name: str,
    wh: tuple[int, int],
    resize_mode: str,
    conf_thr: float,
    nms_iou: float = 0.45,
    save_preview: str | None = None,
) -> list[dict]:
    import cv2

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read --image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x, (off_x, off_y, denom_w, denom_h) = preprocess_like_dut(rgb, wh, resize_mode=resize_mode)
    out = session.run(None, {input_name: x})[0]
    a = np.asarray(out).reshape(-1, np.asarray(out).shape[-1])
    cols = a.shape[-1]
    w_in, h_in = float(wh[0]), float(wh[1])
    needs_unmap = (off_x != 0.0 or off_y != 0.0 or denom_w != w_in or denom_h != h_in) and denom_w > 0 and denom_h > 0

    raw_cands: list[dict] = []
    for row in a:
        if cols == 6:
            cx, cy, bw, bh, conf, cls = [float(v) for v in row[:6]]
            theta = 0.0
        else:
            cx, cy, bw, bh, theta, conf, cls = [float(v) for v in row[:7]]
        if conf < conf_thr:
            continue
        if needs_unmap:
            cx = (cx * w_in - off_x) / denom_w
            cy = (cy * h_in - off_y) / denom_h
            bw = (bw * w_in) / denom_w
            bh = (bh * h_in) / denom_h
        raw_cands.append({"cx": cx, "cy": cy, "w": bw, "h": bh, "theta": theta, "conf": conf, "cls": int(round(cls))})

    raw_cands.sort(key=lambda d: d["conf"], reverse=True)
    # Apply the exact class-aware rotated NMS used on the La Frite board so the printed list
    # matches the cleaned preview image 1-to-1.
    kept: list[dict] = []
    for cand in raw_cands:
        if any(k["cls"] == cand["cls"] and _obb_iou_nms(cand, k) > nms_iou for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= 100:
            break

    print(
        f"\n[+] Real-image check ({image_path}, resize={resize_mode}, conf>={conf_thr}, nms_iou={nms_iou}): "
        f"{len(kept)} detection(s) after NMS (from {len(raw_cands)} raw candidates)"
    )
    for d in kept[:20]:
        name = "person" if d["cls"] == 0 else ("vehicle" if d["cls"] == 1 else f"cls{d['cls']}")
        print(
            f"    {name:7s} conf={d['conf']:.3f} cx={d['cx']:.3f} cy={d['cy']:.3f} "
            f"w={d['w']:.3f} h={d['h']:.3f} theta={math.degrees(d['theta']):+.1f}deg"
        )

    if save_preview:
        vis = bgr.copy()
        ih, iw = vis.shape[:2]
        for d in kept:
            color = (0, 220, 255) if d["cls"] == 0 else (255, 220, 0)
            cx_px, cy_px = d["cx"] * iw, d["cy"] * ih
            bw_px, bh_px = d["w"] * iw, d["h"] * ih
            c, s = math.cos(d["theta"]), math.sin(d["theta"])
            dx, dy = bw_px / 2.0, bh_px / 2.0
            pts = np.array(
                [[int(round(cx_px + px * c - py * s)), int(round(cy_px + px * s + py * c))]
                 for px, py in ((-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy))],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, color, 2, cv2.LINE_AA)
            lbl = f"{'P' if d['cls'] == 0 else 'V'} {d['conf']:.2f}"
            cv2.putText(vis, lbl, (max(2, int(cx_px - dx)), max(14, int(cy_px - dy - 4))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.imwrite(save_preview, vis)
        print(f"[+] Saved post-NMS OBB preview to: {save_preview}")
    return kept


def verify_repo(repo_dir: str = ".", image_path: str | None = None, save_preview: str | None = None) -> bool:
    root = Path(repo_dir).resolve()
    print("=" * 65)
    print("  Weights of Giants - Season Zero Pre-Submission Verifier (wog-cv-v2.1)")
    print("  Target: Libre Computer La Frite (ARM Cortex-A53, 1024MB RAM)")
    print("=" * 65)

    failures: list[str] = []
    warnings: list[str] = []

    # 1. Check manifest (wog.yaml / wog.yml / model.json / pipeline.json)
    manifest_path = None
    for name in ("wog.yaml", "wog.yml", "model.json", "pipeline.json"):
        candidate = root / name
        if candidate.is_file():
            manifest_path = candidate
            break
    if not manifest_path:
        failures.append("Missing 'wog.yaml' or 'model.json' in repository root.")
        return _print_summary(failures, warnings)

    with open(manifest_path, "r", encoding="utf-8") as f:
        if manifest_path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                failures.append("PyYAML is required to parse wog.yaml (pip install pyyaml).")
                return _print_summary(failures, warnings)
            manifest = yaml.safe_load(f)
        else:
            manifest = json.load(f)
    if not isinstance(manifest, dict):
        failures.append(f"Manifest {manifest_path.name} must decode to a JSON/YAML object.")
        return _print_summary(failures, warnings)

    if manifest.get("runtime") != "onnxruntime":
        failures.append("Manifest 'runtime' must be 'onnxruntime' (only supported Season Zero runtime).")
    if manifest.get("trust_remote_code"):
        failures.append("Manifest 'trust_remote_code' is forbidden.")

    domains = manifest.get("domains")
    if domains is not None:
        if not isinstance(domains, list) or not domains or any(d not in ALLOWED_DOMAINS for d in domains):
            failures.append(f"Manifest 'domains' must be a non-empty subset of {sorted(ALLOWED_DOMAINS)}.")

    threads = manifest.get("threads", CORTEX_A53_PROFILE["limits"]["max_threads"])
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= CORTEX_A53_PROFILE["limits"]["max_threads"]:
        failures.append(f"Manifest 'threads' must be an integer 1-{CORTEX_A53_PROFILE['limits']['max_threads']}.")

    frame_policy = manifest.get("frame_policy", {})
    resize_mode = "letterbox"
    if not isinstance(frame_policy, dict):
        failures.append("Manifest 'frame_policy' must be an object.")
    else:
        skip = frame_policy.get("skip", 0)
        if not isinstance(skip, int) or isinstance(skip, bool) or skip != 0:
            failures.append("Manifest 'frame_policy.skip' must be 0 (every clean frame is evaluated).")
        resize_mode = frame_policy.get("resize", "letterbox")
        if resize_mode not in ALLOWED_RESIZE_MODES:
            failures.append(f"Manifest 'frame_policy.resize' must be one of {sorted(ALLOWED_RESIZE_MODES)}.")

    # 2. Check for forbidden file extensions (skip .git/.venv; warn if inside gitignored work/)
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_SCAN_DIRS for part in rel_parts):
            continue
        if p.suffix.lower() in UNSAFE_EXTS:
            if rel_parts and rel_parts[0] in ("work", "runs", "datasets"):
                warnings.append(
                    f"Unsafe checkpoint in local folder '{p.relative_to(root).as_posix()}': "
                    "ensure it is .gitignored and never pushed to GitHub/Hugging Face."
                )
            else:
                failures.append(f"Forbidden unsafe artifact found: {p.relative_to(root).as_posix()}")

    # 3. Check stages and model files
    stages = manifest.get("stages", [])
    if not isinstance(stages, list) or not stages:
        failures.append("No stages defined in manifest.")
        stages = []

    detector_stage = None
    tracker_stage = None
    for s in stages:
        if not isinstance(s, dict):
            continue
        if s.get("type") == "detector" and detector_stage is None:
            detector_stage = s
        elif s.get("type") == "tracker":
            tracker_stage = s
            if s.get("algo") != "bytetrack-lite":
                failures.append("Tracker stage 'algo' must be 'bytetrack-lite'.")

    if not detector_stage:
        failures.append("Manifest must contain at least one stage of type 'detector'.")
        return _print_summary(failures, warnings)

    conf_thr = detector_stage.get("conf_threshold", 0.30)
    if not isinstance(conf_thr, (int, float)) or isinstance(conf_thr, bool) or not 0.0 < float(conf_thr) < 1.0:
        failures.append("Detector 'conf_threshold' must be a float strictly between 0 and 1.")
        conf_thr = 0.30

    iou_thr = detector_stage.get("iou_threshold", 0.45)
    if not isinstance(iou_thr, (int, float)) or isinstance(iou_thr, bool) or not 0.0 < float(iou_thr) < 1.0:
        failures.append("Detector 'iou_threshold' must be a float strictly between 0 and 1.")

    if tracker_stage and isinstance(tracker_stage.get("params"), dict):
        tp = tracker_stage["params"]
        t_thresh = tp.get("track_thresh")
        if t_thresh is not None and isinstance(t_thresh, (int, float)) and float(t_thresh) < float(conf_thr):
            warnings.append(
                f"tracker.params.track_thresh ({t_thresh}) is lower than detector conf_threshold ({conf_thr}); "
                "ByteTrack-Lite stage-2 low-conf recovery works best when track_thresh >= conf_threshold."
            )

    model_rel = detector_stage.get("model")
    if not isinstance(model_rel, str) or not model_rel:
        failures.append("Detector stage is missing 'model' path.")
        return _print_summary(failures, warnings)
    if ".." in model_rel or model_rel.startswith(("/", "\\")) or ":" in model_rel or not model_rel.lower().endswith(".onnx"):
        failures.append("Detector 'model' must be a relative .onnx path inside the repository.")

    model_path = root / model_rel
    if not model_path.exists():
        failures.append(f"Model file not found at: {model_rel}")
        return _print_summary(failures, warnings)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n[+] Manifest:   {manifest_path.name}")
    print(f"[+] Model File: {model_rel} ({model_size_mb:.2f} MB)")

    if model_size_mb > CORTEX_A53_PROFILE["limits"]["artifact_mb"]:
        failures.append(
            f"Model size {model_size_mb:.2f} MB exceeds Low-Energy Edge limit "
            f"({CORTEX_A53_PROFILE['limits']['artifact_mb']} MB)."
        )

    raw_size = detector_stage.get("input_size")
    max_side = CORTEX_A53_PROFILE["limits"]["max_input_side_px"]
    if (
        not isinstance(raw_size, list)
        or len(raw_size) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) and 32 <= v <= max_side for v in raw_size)
    ):
        failures.append(f"Detector 'input_size' must be [w, h] with each side 32-{max_side}.")
        return _print_summary(failures, warnings)
    input_w, input_h = raw_size
    if max(input_w, input_h) > 416:
        warnings.append(
            f"Input resolution {input_w}x{input_h} is large for a 1.2 GHz Cortex-A53 CPU; "
            "verify your model stays above the 2.4 FPS floor (320x320 or 416x416 is recommended)."
        )

    # Optional ONNX IR/opset check (install `onnx` locally for full parity)
    try:
        import onnx

        onnx_model = onnx.load(str(model_path), load_external_data=False)
        ir_ver = int(getattr(onnx_model, "ir_version", 0) or 0)
        if ir_ver <= 0 or ir_ver > CORTEX_A53_PROFILE["limits"]["max_onnx_ir"]:
            failures.append(
                f"ONNX IR version {ir_ver} exceeds target board ceiling (IR <= {CORTEX_A53_PROFILE['limits']['max_onnx_ir']})."
            )
        opsets = [
            int(imp.version)
            for imp in getattr(onnx_model, "opset_import", [])
            if getattr(imp, "domain", "") in ("", "ai.onnx")
        ]
        if not opsets or max(opsets) > CORTEX_A53_PROFILE["limits"]["max_onnx_opset"]:
            failures.append(
                f"ONNX opset {opsets or 'missing'} exceeds ceiling (opset <= {CORTEX_A53_PROFILE['limits']['max_onnx_opset']})."
            )
    except ImportError:
        warnings.append("Package 'onnx' not installed locally; skipping ONNX IR/opset ceiling check.")

    # 4. Run ONNX Runtime Inference
    print(f"[+] Input:      [1, 3, {input_h}, {input_w}] float32 RGB [0, 1] (resize={resize_mode})")
    print(f"[+] Thresholds: conf={conf_thr}, nms_iou={iou_thr}, threads={threads}")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), sess_options, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]

    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 3, input_h, input_w)).astype(np.float32)

    print("[+] Warming up session...")
    for _ in range(5):
        session.run(None, {input_meta.name: dummy_input})

    print("[+] Running 30-frame local CPU latency smoke test...")
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_meta.name: dummy_input})
        latencies.append(time.perf_counter() - t0)

    mean_sec = float(np.mean(latencies))
    fps = 1.0 / mean_sec if mean_sec > 0 else 0.0
    p95_sec = float(np.percentile(latencies, 95))

    print("\nLocal Host CPU Smoke Benchmark (NOT equivalent to 1.2 GHz Cortex-A53 La Frite):")
    print(f"  Mean Latency: {mean_sec * 1000.0:.1f} ms")
    print(f"  P95 Latency:  {p95_sec * 1000.0:.1f} ms")
    print(f"  Host CPU FPS: {fps:.1f} FPS (official board floor >= 2.4 FPS)")

    detections = outputs[0]
    print(f"  Output Tensor Shape: {detections.shape}")
    val_fail, val_warn = _validate_output_values(detections)
    failures.extend(val_fail)
    warnings.extend(val_warn)
    if not val_fail:
        print("  Contract Output Format: VALID ([cx, cy, w, h, theta_rad, conf, cls, ...])")

    if image_path:
        _decode_and_draw_preview(
            image_path,
            session,
            input_meta.name,
            (input_w, input_h),
            resize_mode,
            float(conf_thr),
            nms_iou=float(iou_thr),
            save_preview=save_preview,
        )

    # Check for open-source LICENSE file (and warn about Ultralytics AGPL-3.0 obligations)
    has_license = any((root / name).is_file() for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"))
    if not has_license:
        warnings.append(
            "No LICENSE file found at repo root. Add an approved SPDX license before submitting "
            "(IMPORTANT: any model trained/exported from Ultralytics YOLO11/YOLOv8 MUST be licensed AGPL-3.0; "
            "for Apache-2.0/MIT start from YOLOX, NanoDet-Plus, or PP-PicoDet)."
        )

    print("\nMeasured La Frite Cortex-A53 Reference Speeds (for sizing your model vs. 2.4 FPS floor):")
    print("  NanoDet-Plus-m  (320x320,  4.58 MB): 4.68 FPS · 276.5 MB RAM · 48.61% mAP · 43.20% HOTA")
    print("  MobileNetV3-SSD (320x320, 13.21 MB): 3.94 FPS · 274.8 MB RAM · 28.30% mAP · 31.73% HOTA")
    print("  YOLOX-Nano      (416x416,  3.52 MB): 3.37 FPS · 277.6 MB RAM · 48.53% mAP · 44.66% HOTA")
    print("  PP-PicoDet-S    (320x320,  4.56 MB): 3.26 FPS · 275.3 MB RAM · 49.04% mAP · 43.79% HOTA")

    if fps < CORTEX_A53_PROFILE["limits"]["fps_floor"]:
        failures.append(f"Host FPS {fps:.1f} is already below the {CORTEX_A53_PROFILE['limits']['fps_floor']} FPS floor.")
    else:
        warnings.append(
            "Host CPU FPS is NOT comparable to the 1.2 GHz Cortex-A53 La Frite board (often 50x-200x faster on desktop CPUs); "
            "compare your architecture, input size, and MB against the measured La Frite baselines above, "
            "and run `python score_local.py` to estimate accuracy before spending a Run Token."
        )

    return _print_summary(failures, warnings)


def _print_summary(failures: list[str], warnings: list[str]) -> bool:
    print("\n" + "=" * 65)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  [!] {w}")
    if failures:
        print("\nVerification FAILED:")
        for f in failures:
            print(f"  [X] {f}")
        print("\nPlease fix the above issues before submitting your model.")
        return False
    print("Verification PASSED!")
    print("Your model meets the Low-Energy Edge Challenge packaging constraints.")
    print("You can submit this repository via GitHub or Hugging Face.")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a Weights of Giants Season Zero submission repo locally.")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent), help="Repository directory to verify")
    ap.add_argument("--image", default=None, help="Optional sample image to test exact La Frite preprocess + decode")
    ap.add_argument("--save-preview", default=None, help="Optional output path to draw decoded OBB boxes on --image")
    args = ap.parse_args(argv)
    return 0 if verify_repo(args.repo, image_path=args.image, save_preview=args.save_preview) else 1


if __name__ == "__main__":
    raise SystemExit(main())
