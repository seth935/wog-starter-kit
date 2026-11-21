"""Weights of Giants — Local Pre-Submission Accuracy & Tracking Scorer (wog-cv-v2.1).

Evaluates your repository's ONNX model (`model.json` / `wog.yaml`) on the included
non-test-suite development clip (`dev_clip/`, 25 frames @ 3.75 FPS, pedestrians + vehicles)
plus a blank empty-scene canary check, using the exact La Frite `dut_shim.py` pipeline
(letterbox/stretch/crop -> ONNX -> coordinate unmap -> rotated NMS -> ByteTrack-Lite)
and the exact `wog-cv-v2.1` rotated OBB mAP@50:95 + native HOTA scorer.

Usage:
    python score_local.py
    python score_local.py --clip path/to/your_val_clip --save-vis work/dev_vis.jpg
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Real hardware measurements on Libre Computer La Frite (Cortex-A53 @ 1.2 GHz, 1 GB RAM)
# under wog-cv-v2.1 across all 15 Season Zero clips, plus measured scores on this dev_clip.
# (Model, Input, Size, Dev mAP, Dev HOTA, Official mAP, Official HOTA, Board FPS, RAM, Score, License)
LA_FRITE_BASELINES = [
    ("NanoDet-Plus-m", "320x320", "4.58 MB", "18.31%", "18.88%", "48.61%", "43.20%", "4.68 FPS", "276.5 MB", "48.48", "Apache-2.0"),
    ("YOLOX-Nano", "416x416", "3.52 MB", "17.55%", "19.47%", "48.53%", "44.66%", "3.37 FPS", "277.6 MB", "45.00", "Apache-2.0"),
    ("PP-PicoDet-S", "320x320", "4.56 MB", "20.48%", "19.57%", "49.04%", "43.79%", "3.26 FPS", "275.3 MB", "44.96", "Apache-2.0"),
    ("MobileNetV3-SSD", "320x320", "13.21 MB", "3.33%", "8.13%", "28.30%", "31.73%", "3.94 FPS", "274.8 MB", "36.51*", "BSD-3-Clause"),
]

REF_W, REF_H = 480.0, 320.0
MIN_REQUIRED_LONG_PX = 15.0
MIN_REQUIRED_SHORT_PX = 10.0


def _obb_iou(b1: tuple[float, float, float, float, float], b2: tuple[float, float, float, float, float]) -> float:
    """Exact rotated OBB IoU via OpenCV rotatedRectangleIntersection."""
    cx1, cy1, w1, h1, th1 = b1
    cx2, cy2, w2, h2, th2 = b2
    if w1 <= 1e-7 or h1 <= 1e-7 or w2 <= 1e-7 or h2 <= 1e-7:
        return 0.0
    if abs(th1) < 1e-4 and abs(th2) < 1e-4:
        ix1 = max(cx1 - w1 * 0.5, cx2 - w2 * 0.5)
        iy1 = max(cy1 - h1 * 0.5, cy2 - h2 * 0.5)
        ix2 = min(cx1 + w1 * 0.5, cx2 + w2 * 0.5)
        iy2 = min(cy1 + h1 * 0.5, cy2 + h2 * 0.5)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = w1 * h1 + w2 * h2 - inter
        return float(inter / union) if union > 1e-9 else 0.0
    r1 = ((float(cx1), float(cy1)), (float(w1), float(h1)), float(math.degrees(th1)))
    r2 = ((float(cx2), float(cy2)), (float(w2), float(h2)), float(math.degrees(th2)))
    kind, pts = cv2.rotatedRectangleIntersection(r1, r2)
    if kind == cv2.INTERSECT_NONE or pts is None or len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts.reshape(-1, 2).astype(np.float32))
    inter = float(cv2.contourArea(hull))
    union = w1 * h1 + w2 * h2 - inter
    return float(max(0.0, min(1.0, inter / union))) if union > 1e-9 else 0.0


def _box_of(rec: dict) -> tuple[float, float, float, float, float]:
    bb = rec["bbox"]
    return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]), float(rec.get("theta", 0.0)))


def _is_optional_gt(gt: dict, all_gts: list[dict], ignore_regions: list[list[float]] | None = None) -> bool:
    """Mirror src/shared/ignore_zones/logic.py required vs. optional GT classification."""
    if gt.get("ignore"):
        return True
    cx, cy, bw, bh, _ = _box_of(gt)
    w_px, h_px = bw * REF_W, bh * REF_H
    long_px, short_px = max(w_px, h_px), min(w_px, h_px)
    area_px = w_px * h_px

    # 1. Explicit clip ignore_regions (with full-vehicle guard so large trucks stay required)
    if ignore_regions:
        x1, y1, x2, y2 = cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5
        box_a = max(1e-9, bw * bh)
        for rx1, ry1, rx2, ry2 in ignore_regions:
            inter = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(0.0, min(y2, ry2) - max(y1, ry1))
            if inter / box_a >= 0.55 and (area_px <= 440.0 or (short_px <= 14.0 and area_px <= 750.0)):
                return True

    # 2. Sub-15px far-field optional tier
    if long_px < MIN_REQUIRED_LONG_PX or short_px < MIN_REQUIRED_SHORT_PX:
        return True

    # 3. Small (<48px) frame-edge truncated target
    x1, y1, x2, y2 = cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5
    touches_edge = (x1 <= 0.008) or (y1 <= 0.008) or (x2 >= 0.992) or (y2 >= 0.992)
    if touches_edge and long_px < 48.0:
        return True

    # 4. Heavily occluded (>=50% covered by nearer object with lower y2)
    box_a = max(1e-9, bw * bh)
    for other in all_gts:
        if other is gt:
            continue
        ocx, ocy, obw, obh, _ = _box_of(other)
        ox1, oy1, ox2, oy2 = ocx - obw * 0.5, ocy - obh * 0.5, ocx + obw * 0.5, ocy + obh * 0.5
        if oy2 <= y2 + 0.005:
            continue
        inter = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(0.0, min(y2, oy2) - max(y1, oy1))
        if inter / box_a >= 0.50:
            return True
    return False


def _rotated_nms(dets: list[list], iou_thr: float = 0.45, max_dets: int = 100) -> list[list]:
    """Class-aware rotated NMS matching backend/rig/dut_shim.py."""
    ordered = sorted(dets, key=lambda r: r[5], reverse=True)
    kept: list[list] = []
    for cand in ordered:
        cb = (cand[0], cand[1], cand[2], cand[3], cand[4])
        if any(k[6] == cand[6] and _obb_iou(cb, (k[0], k[1], k[2], k[3], k[4])) > iou_thr for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= max_dets:
            break
    return kept


class ByteTrackLite:
    """Exact mirror of backend/rig/dut_shim.py Mode A ByteTrack-Lite tracker."""

    def __init__(self, track_thresh: float = 0.35, track_buffer: int = 6, match_thresh: float = 0.80):
        self.track_thresh = float(track_thresh)
        self.track_buffer = max(1, int(track_buffer))
        self.min_iou = max(0.10, min(0.80, 1.0 - float(match_thresh)))
        self._next_tid = 1
        self._tracks: list[dict] = []

    def _greedy_match(self, trk_indices: list[int], det_indices: list[int], dets: list[list], min_iou: float):
        pairs: list[tuple[float, int, int]] = []
        for ti in trk_indices:
            trk = self._tracks[ti]
            dt = float(trk["lost"] + 1)
            px = trk["box"][0] + trk["vx"] * dt
            py = trk["box"][1] + trk["vy"] * dt
            pred_box = (px, py, trk["box"][2], trk["box"][3], trk["box"][4])
            for di in det_indices:
                det = dets[di]
                if det[6] != trk["cls"]:
                    continue
                ov = _obb_iou(pred_box, (det[0], det[1], det[2], det[3], det[4]))
                if ov >= min_iou:
                    pairs.append((ov, ti, di))
        pairs.sort(key=lambda x: x[0], reverse=True)
        used_t, used_d, matches = set(), set(), []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))
        return matches, used_t, used_d

    def update(self, dets: list[list]) -> list[list]:
        if not dets:
            for trk in self._tracks:
                trk["lost"] += 1
            self._tracks = [t for t in self._tracks if t["lost"] <= self.track_buffer]
            return dets
        high_idx = [i for i, d in enumerate(dets) if d[7] is None and d[5] >= self.track_thresh]
        low_idx = [i for i, d in enumerate(dets) if d[7] is None and d[5] < self.track_thresh]
        if not high_idx and not low_idx:
            return dets
        all_t = list(range(len(self._tracks)))
        m1, used_t1, used_d1 = self._greedy_match(all_t, high_idx, dets, self.min_iou)
        rem_t = [ti for ti in all_t if ti not in used_t1 and self._tracks[ti]["lost"] <= 2]
        m2, used_t2, used_d2 = self._greedy_match(rem_t, low_idx, dets, max(0.30, self.min_iou))
        for ti, di in m1 + m2:
            trk, det = self._tracks[ti], dets[di]
            dt = max(1, trk["lost"] + 1)
            trk["vx"] = 0.6 * trk["vx"] + 0.4 * ((det[0] - trk["box"][0]) / float(dt))
            trk["vy"] = 0.6 * trk["vy"] + 0.4 * ((det[1] - trk["box"][1]) / float(dt))
            trk["box"] = det[:5]
            trk["lost"] = 0
            det[7] = trk["tid"]
        matched_t = used_t1 | used_t2
        for ti in all_t:
            if ti not in matched_t:
                self._tracks[ti]["lost"] += 1
        for di in high_idx:
            if di not in used_d1:
                det = dets[di]
                tid = self._next_tid
                self._next_tid += 1
                det[7] = tid
                self._tracks.append({"tid": tid, "box": det[:5], "vx": 0.0, "vy": 0.0, "cls": det[6], "lost": 0})
        unmatched_low = set(low_idx) - used_d2
        if unmatched_low:
            dets[:] = [d for i, d in enumerate(dets) if i not in unmatched_low]
        self._tracks = [t for t in self._tracks if t["lost"] <= self.track_buffer]
        return dets


def _load_manifest(repo_root: Path) -> tuple[dict, Path]:
    for name in ("wog.yaml", "wog.yml", "model.json", "pipeline.json"):
        p = repo_root / name
        if p.is_file():
            if p.suffix.lower() in (".yaml", ".yml"):
                import yaml

                return yaml.safe_load(p.read_text(encoding="utf-8")), p
            return json.loads(p.read_text(encoding="utf-8")), p
    raise FileNotFoundError(f"No model.json or wog.yaml found in {repo_root}")


def _load_clip(clip_dir: Path) -> tuple[list[np.ndarray], list[list[dict]], dict]:
    meta_p = clip_dir / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    frame_paths = sorted(glob.glob(str(clip_dir / "frames" / "frame_*.*")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames/frame_*.* found under {clip_dir}")
    frames_rgb: list[np.ndarray] = []
    gt_frames: list[list[dict]] = []
    for fp in frame_paths:
        bgr = cv2.imread(fp)
        if bgr is None:
            continue
        frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        stem = Path(fp).stem
        lp = clip_dir / "labels" / f"{stem}.txt"
        f_gt: list[dict] = []
        if lp.is_file():
            for ln in lp.read_text(encoding="utf-8").splitlines():
                p = ln.strip().split()
                if len(p) >= 5:
                    f_gt.append({
                        "cls": int(float(p[0])),
                        "bbox": [float(p[1]), float(p[2]), float(p[3]), float(p[4])],
                        "theta": float(p[5]) if len(p) >= 6 else 0.0,
                        "id": int(float(p[6])) if len(p) >= 7 else None,
                        "ignore": bool(int(float(p[7]))) if len(p) >= 8 else False,
                    })
        gt_frames.append(f_gt)
    return frames_rgb, gt_frames, meta


def _preprocess(frame_rgb: np.ndarray, wh: tuple[int, int], resize_mode: str):
    w, h = wh
    orig_h, orig_w = frame_rgb.shape[:2]
    tf = (0.0, 0.0, float(w), float(h))
    if orig_h != h or orig_w != w:
        if resize_mode == "letterbox":
            r = min(w / float(orig_w), h / float(orig_h))
            nw, nh = max(1, int(round(orig_w * r))), max(1, int(round(orig_h * r)))
            resized = cv2.resize(frame_rgb, (nw, nh))
            canvas = np.full((h, w, 3), 114, dtype=np.uint8)
            px, py = (w - nw) // 2, (h - nh) // 2
            canvas[py : py + nh, px : px + nw] = resized
            frame_rgb = canvas
            tf = (float(px), float(py), float(nw), float(nh))
        elif resize_mode == "crop":
            r = max(w / float(orig_w), h / float(orig_h))
            sw, sh = max(w, int(round(orig_w * r))), max(h, int(round(orig_h * r)))
            resized = cv2.resize(frame_rgb, (sw, sh))
            cx, cy = (sw - w) // 2, (sh - h) // 2
            frame_rgb = resized[cy : cy + h, cx : cx + w]
            tf = (-float(cx), -float(cy), float(sw), float(sh))
        else:
            frame_rgb = cv2.resize(frame_rgb, (w, h))
    x = (frame_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(x), tf


def _score_ap_at_iou(pred_frames: list[list[dict]], gt_frames: list[list[dict]], iou_thr: float, ignore_regions: list) -> float:
    """Compute exact 101-point pooled PR-curve AP at iou_thr matching backend/rig/pi_scorer.py."""
    total_gt = 0
    scored_dets: list[tuple[float, int]] = []
    for fi, gts in enumerate(gt_frames):
        req_f = [g for g in gts if not _is_optional_gt(g, gts, ignore_regions)]
        opt_f = [g for g in gts if _is_optional_gt(g, gts, ignore_regions)]
        total_gt += len(req_f)
        preds = sorted(pred_frames[fi], key=lambda p: float(p["conf"]), reverse=True)
        used_req: set[int] = set()
        used_opt: set[int] = set()
        for p in preds:
            pb, pcls, pconf = _box_of(p), int(p["cls"]), float(p["conf"])
            best_ov, best_j = iou_thr, -1
            for j, g in enumerate(req_f):
                if int(g["cls"]) != pcls or j in used_req:
                    continue
                ov = _obb_iou(pb, _box_of(g))
                if ov >= best_ov:
                    best_ov, best_j = ov, j
            if best_j >= 0:
                used_req.add(best_j)
                scored_dets.append((pconf, 1))
                continue
            best_o, best_oj = iou_thr, -1
            for j, g in enumerate(opt_f):
                if int(g["cls"]) != pcls or j in used_opt:
                    continue
                ov = _obb_iou(pb, _box_of(g))
                if ov >= best_o:
                    best_o, best_oj = ov, j
            if best_oj >= 0:
                used_opt.add(best_oj)
                continue
            scored_dets.append((pconf, 0))

    if total_gt == 0 or not scored_dets:
        return 0.0
    scored_dets.sort(key=lambda x: x[0], reverse=True)
    tp_cum, fp_cum = 0, 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _, is_tp in scored_dets:
        if is_tp:
            tp_cum += 1
        else:
            fp_cum += 1
        precisions.append(tp_cum / float(tp_cum + fp_cum))
        recalls.append(tp_cum / float(total_gt))
    for i in range(len(precisions) - 2, -1, -1):
        if precisions[i] < precisions[i + 1]:
            precisions[i] = precisions[i + 1]
    ap_sum = 0.0
    for r_step in range(101):
        r_thr = r_step / 100.0
        p_val = next((precisions[i] for i, rec in enumerate(recalls) if rec >= r_thr), 0.0)
        ap_sum += p_val
    return (ap_sum / 101.0) * 100.0


def _score_hota(pred_frames: list[list[dict]], gt_frames: list[list[dict]], ignore_regions: list) -> dict[str, float]:
    """Compute native HOTA (sqrt(DetA * AssA)) across IoU 0.50..0.95 with MOT-style distractor filtering."""
    hota_vals, deta_vals, assa_vals = [], [], []
    for k in range(10):
        thr = round(0.50 + 0.05 * k, 2)
        tp_tot, fp_tot, fn_tot = 0, 0, 0
        gt_tot: dict[tuple, int] = defaultdict(int)
        pr_tot: dict[tuple, int] = defaultdict(int)
        pair_tot: dict[tuple, int] = defaultdict(int)
        matched_pairs: list[tuple[tuple, tuple]] = []

        for fi, gts in enumerate(gt_frames):
            req = [g for g in gts if not _is_optional_gt(g, gts, ignore_regions)]
            opt = [g for g in gts if _is_optional_gt(g, gts, ignore_regions)]
            preds = sorted(pred_frames[fi], key=lambda p: float(p["conf"]), reverse=True)
            used_r, used_o = set(), set()
            for p in preds:
                pb, pcls = _box_of(p), int(p["cls"])
                best_ov, best_j = thr, -1
                for j, g in enumerate(req):
                    if int(g["cls"]) != pcls or j in used_r:
                        continue
                    ov = _obb_iou(pb, _box_of(g))
                    if ov >= best_ov:
                        best_ov, best_j = ov, j
                if best_j >= 0:
                    used_r.add(best_j)
                    tp_tot += 1
                    gk = (pcls, req[best_j].get("id"))
                    pk = (pcls, p.get("id"))
                    gt_tot[gk] += 1
                    pr_tot[pk] += 1
                    pair_tot[(gk, pk)] += 1
                    matched_pairs.append((gk, pk))
                    continue
                # MOT distractor filter: drop detections matching optional/ignored GT
                best_o, best_oj = thr, -1
                for j, g in enumerate(opt):
                    if int(g["cls"]) != pcls or j in used_o:
                        continue
                    ov = _obb_iou(pb, _box_of(g))
                    if ov >= best_o:
                        best_o, best_oj = ov, j
                if best_oj >= 0:
                    used_o.add(best_oj)
                    continue
                fp_tot += 1
                pr_tot[(pcls, p.get("id"))] += 1
            for j, g in enumerate(req):
                if j not in used_r:
                    fn_tot += 1
                    gt_tot[(int(g["cls"]), g.get("id"))] += 1

        denom = tp_tot + fp_tot + fn_tot
        deta = (tp_tot / float(denom)) if denom > 0 else 0.0
        if matched_pairs:
            ass_sum = 0.0
            for gk, pk in matched_pairs:
                tpa = pair_tot[(gk, pk)]
                ass_denom = gt_tot[gk] + pr_tot[pk] - tpa
                ass_sum += (tpa / float(ass_denom)) if ass_denom > 0 else 0.0
            assa = ass_sum / float(len(matched_pairs))
        else:
            assa = 0.0
        deta_vals.append(deta * 100.0)
        assa_vals.append(assa * 100.0)
        hota_vals.append(math.sqrt(max(0.0, deta * assa)) * 100.0)
    return {
        "hota": round(sum(hota_vals) / len(hota_vals), 2),
        "det_a": round(sum(deta_vals) / len(deta_vals), 2),
        "ass_a": round(sum(assa_vals) / len(assa_vals), 2),
    }


def score_repo(repo_dir: str, clip_dir: str | None = None, save_vis: str | None = None) -> dict:
    root = Path(repo_dir).resolve()
    manifest, manifest_path = _load_manifest(root)
    det_stage = next((s for s in manifest.get("stages", []) if s.get("type") == "detector"), None)
    trk_stage = next((s for s in manifest.get("stages", []) if s.get("type") == "tracker"), None)
    if not det_stage:
        raise RuntimeError("Manifest has no detector stage.")

    model_path = root / det_stage["model"]
    wh = (int(det_stage["input_size"][0]), int(det_stage["input_size"][1]))
    conf_thr = float(det_stage.get("conf_threshold", 0.30))
    nms_iou = float(det_stage.get("iou_threshold", 0.45))
    resize_mode = str((manifest.get("frame_policy") or {}).get("resize", "letterbox"))
    threads = int(manifest.get("threads", 4))
    tp = (trk_stage or {}).get("params") or {}

    clip_path = Path(clip_dir).resolve() if clip_dir else (root / "dev_clip")
    frames_rgb, gt_frames, meta = _load_clip(clip_path)
    ignore_regions = meta.get("ignore_regions") or []

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    tracker = ByteTrackLite(
        track_thresh=float(tp.get("track_thresh", conf_thr)),
        track_buffer=int(tp.get("track_buffer", 6)),
        match_thresh=float(tp.get("match_thresh", 0.80)),
    )

    def _run_frame(frame_rgb: np.ndarray, update_tracker: bool = True) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        x, (ox, oy, dw, dh) = _preprocess(frame_rgb, wh, resize_mode)
        out = sess.run(None, {inp_name: x})[0]
        a = np.asarray(out).reshape(-1, np.asarray(out).shape[-1])
        cols = a.shape[-1]
        win, hin = float(wh[0]), float(wh[1])
        needs_unmap = (ox != 0.0 or oy != 0.0 or dw != win or dh != hin) and dw > 0 and dh > 0
        cands: list[list] = []
        for row in a:
            if cols == 6:
                cx, cy, bw, bh, cval, cls = [float(v) for v in row[:6]]
                th, tid = 0.0, None
            else:
                cx, cy, bw, bh, th, cval, cls = [float(v) for v in row[:7]]
                raw_tid = int(round(float(row[7]))) if cols == 8 else 0
                tid = raw_tid if raw_tid > 0 else None
            if cval < conf_thr:
                continue
            if needs_unmap:
                cx = (cx * win - ox) / dw
                cy = (cy * hin - oy) / dh
                bw = (bw * win) / dw
                bh = (bh * hin) / dh
            cands.append([cx, cy, bw, bh, th, cval, int(round(cls)), tid])
        kept = _rotated_nms(cands, iou_thr=nms_iou, max_dets=100)
        if update_tracker:
            tracker.update(kept)
        ms = (time.perf_counter() - t0) * 1000.0
        return (
            [{"bbox": d[:4], "theta": d[4], "conf": d[5], "cls": d[6], "id": d[7]} for d in kept],
            ms,
        )

    pred_frames: list[list[dict]] = []
    times_ms: list[float] = []
    for fr in frames_rgb:
        pf, ms = _run_frame(fr, update_tracker=True)
        pred_frames.append(pf)
        times_ms.append(ms)

    # 10-frame blank/low-noise canary check (tests empty-scene false alarms)
    rng = np.random.RandomState(42)
    canary_quiet = 0
    canary_fps = 0
    for k in range(10):
        blank = np.clip(110 + k * 4 + rng.normal(0, 3, (320, 480, 3)), 0, 255).astype(np.uint8)
        cp, _ = _run_frame(blank, update_tracker=False)
        canary_fps += len(cp)
        if not cp:
            canary_quiet += 1

    ap_by_iou = [
        _score_ap_at_iou(pred_frames, gt_frames, round(0.50 + 0.05 * k, 2), ignore_regions)
        for k in range(10)
    ]
    map_50_95 = round(sum(ap_by_iou) / len(ap_by_iou), 2)
    map50 = round(ap_by_iou[0], 2)
    map75 = round(ap_by_iou[5], 2)
    hota_res = _score_hota(pred_frames, gt_frames, ignore_regions)
    mean_ms = round(sum(times_ms) / len(times_ms), 2) if times_ms else 0.0

    if save_vis and frames_rgb:
        mid = len(frames_rgb) // 2
        vis = cv2.cvtColor(frames_rgb[mid], cv2.COLOR_RGB2BGR)
        ih, iw = vis.shape[:2]
        for g in gt_frames[mid]:
            cx, cy, bw, bh, _ = _box_of(g)
            cv2.rectangle(vis, (int((cx - bw / 2) * iw), int((cy - bh / 2) * ih)),
                          (int((cx + bw / 2) * iw), int((cy + bh / 2) * ih)), (90, 200, 90), 1)
        for p in pred_frames[mid]:
            cx, cy, bw, bh, _ = _box_of(p)
            col = (0, 220, 255) if p["cls"] == 0 else (255, 220, 0)
            cv2.rectangle(vis, (int((cx - bw / 2) * iw), int((cy - bh / 2) * ih)),
                          (int((cx + bw / 2) * iw), int((cy + bh / 2) * ih)), col, 2)
        Path(save_vis).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_vis, vis)

    print("=" * 72)
    print("  Weights of Giants — Local Dev-Clip Score (wog-cv-v2.1 Scorer)")
    print("=" * 72)
    print(f"  Manifest:    {manifest_path.name} ({det_stage['model']}, {wh[0]}x{wh[1]}, {resize_mode})")
    print(f"  Thresholds:  conf={conf_thr}, nms_iou={nms_iou}, track_buffer={tracker.track_buffer}")
    print(f"  Dev Clip:    {meta.get('test_id', clip_path.name)} ({len(frames_rgb)} frames @ {meta.get('fps', '?')} FPS — NOT in test suite)")
    print("-" * 92)
    print(f"  Detection mAP@50:95 :  {map_50_95:6.2f}%   (passing baselines score 17.6%–20.5% on this hard dev clip)")
    print(f"  Detection mAP@50    :  {map50:6.2f}%   (mAP@75: {map75:.2f}%)")
    print(f"  Tracking HOTA       :  {hota_res['hota']:6.2f}%   (DetA {hota_res['det_a']:.1f}% · AssA {hota_res['ass_a']:.1f}% | baselines: 18.9%–19.6%)")
    print(f"  Blank Canary Quiet  :  {canary_quiet}/10 frames ({canary_fps} false alarms)")
    print(f"  Host CPU Latency    :  {mean_ms:.1f} ms/frame (NOT board speed — see La Frite table below)")
    print("-" * 92)
    print("  Season Zero Reference Baselines (Dev Clip vs. Full 15-Clip Physical La Frite Hardware):")
    print(f"  {'Model':<16} {'Input':<7} {'Size':<8} {'Dev mAP':>7} {'Dev HOTA':>8} | {'Off mAP':>7} {'Off HOTA':>8} {'Board FPS':>9} {'RAM':>8} {'Score':>6} {'License'}")
    for row in LA_FRITE_BASELINES:
        print(f"  {row[0]:<16} {row[1]:<7} {row[2]:<8} {row[3]:>7} {row[4]:>8} | {row[5]:>7} {row[6]:>8} {row[7]:>9} {row[8]:>8} {row[9]:>6} {row[10]}")
    print("  * MobileNetV3-SSD failed rural eligibility gates (rural mAP 17.4 < 25, HOTA 21.2 < 30, domain 28.2 < 35).")
    print("-" * 92)
    if map_50_95 < 10.0 or hota_res["hota"] < 12.0:
        print("  [!] Low dev-clip accuracy compared to passing Season Zero baselines (~17–19% mAP / ~18–19% HOTA here).")
        print("      Check RGB/letterbox normalization, conf_threshold, and class remap before spending a Run Token.")
    else:
        print("  [+] Competitive with passing Season Zero baselines on this dev clip — run verify_local.py before submitting!")
    if save_vis:
        print(f"  [+] Saved GT (green) vs. Pred (cyan/yellow) frame preview -> {save_vis}")
    print("=" * 92)

    return {
        "map_50_95": map_50_95,
        "map50": map50,
        "map75": map75,
        **hota_res,
        "canary_quiet": canary_quiet,
        "canary_fps": canary_fps,
        "host_ms": mean_ms,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Locally score a WOG submission on the non-test dev clip.")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent), help="Repository directory")
    ap.add_argument("--clip", default=None, help="Optional path to custom labeled clip directory")
    ap.add_argument("--save-vis", default=None, help="Optional path to save GT-vs-Pred overlay frame")
    args = ap.parse_args(argv)
    score_repo(args.repo, clip_dir=args.clip, save_vis=args.save_vis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
