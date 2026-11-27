"""Weights of Giants — Permissive ONNX Contract Wrapper (wog-cv-v2.1).

LICENSE POLICY (Permissive Open-Source Only):
  Season Zero accepts permissive SPDX licenses ONLY (Apache-2.0, MIT, BSD-2/3-Clause,
  ISC, 0BSD, Unlicense, CC0-1.0, Zlib, BSL-1.0, CC-BY-4.0).
  Copyleft licenses (AGPL-3.0, GPL, LGPL, MPL-2.0, EPL-2.0, CC-BY-SA) — including
  Ultralytics YOLO11 / YOLOv8 derivatives — are NOT accepted.
  Start from permissive architectures such as YOLOX-Nano (Apache-2.0),
  NanoDet-Plus (Apache-2.0), PP-PicoDet-S (Apache-2.0), or custom PyTorch/TorchVision models.

Wraps a raw detection ONNX into the exact La Frite ONNX Runtime contract:

    Input:  images     [1, 3, H, W]  float32, RGB in [0.0, 1.0]
    Output: detections [1, K, 7]     float32, [cx, cy, w, h, theta_rad, conf, cls]
            - cx, cy, w, h normalized to [0, 1] relative to [W, H]
            - theta_rad in radians (0.0 for upright AABB detectors)
            - conf in [0, 1]
            - cls: 0 = person, 1 = vehicle

Supported input head layouts:
  1. --mode wrap-yolox  Apache-2.0 Megvii YOLOX raw ONNX [1, N, 5+C]
                        ([cx, cy, w, h, obj_conf, cls_0..cls_{C-1}], conf = obj_conf * cls_prob;
                        automatically bakes RGB [0,1] -> BGR [0,255] + [8,16,32] grid decoding)
  2. --mode wrap-onnx   Channel-first anchor-free raw ONNX [1, 4+C, N] (AABB) or [1, 5+C, N] (OBB)

Examples:
    # 1. Wrap official Apache-2.0 YOLOX-Nano ONNX (auto-remaps COCO + bakes BGRx255 & grid decode):
    python export_onnx.py --mode wrap-yolox --onnx work/yolox_nano.onnx --imgsz 416 --coco-remap --out models/detector.onnx --update-manifest

    # 2. Wrap a custom 2-class YOLOX ONNX
    python export_onnx.py --mode wrap-yolox --onnx work/yolox_2class.onnx --imgsz 320 --num-classes 2 --out models/detector.onnx --update-manifest

    # 3. Wrap a channel-first [1, 4+C, N] AABB or [1, 5+C, N] OBB raw ONNX
    python export_onnx.py --mode wrap-onnx --onnx work/raw_detector.onnx --imgsz 320 --num-classes 2 --out models/detector.onnx --update-manifest
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# COCO 80-class indices -> WOG Season Zero classes:
#   0 = person
#   1 = vehicle (bicycle=1, car=2, motorcycle=3, bus=5, truck=7)
COCO_PERSON_INDICES = [0]
COCO_VEHICLE_INDICES = [1, 2, 3, 5, 7]


def _get_main_opset(model: onnx.ModelProto) -> int:
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return int(imp.version)
    return 17


INPUT_FORMATS = ("rgb01", "bgr01", "rgb255", "bgr255")


def _prepend_input_preprocess(model: onnx.ModelProto, input_format: str, prefix: str) -> None:
    """Convert La Frite `images` (RGB [0, 1]) into the inner model's expected channel order/scale."""
    if input_format not in INPUT_FORMATS:
        raise ValueError(f"Unsupported input_format={input_format!r}; choose one of {INPUT_FORMATS}.")
    if input_format == "rgb01":
        return

    graph = model.graph
    if not graph.input:
        raise ValueError("ONNX model has no graph inputs.")
    orig_inp = graph.input[0]
    orig_name = orig_inp.name
    target_name = f"{prefix}pre_target_{orig_name}"

    for node in graph.node:
        for idx, inp_name in enumerate(node.input):
            if inp_name == orig_name:
                node.input[idx] = target_name

    orig_inp.name = "images"
    pre_inits: list[onnx.TensorProto] = []
    pre_nodes: list[onnx.NodeProto] = []
    cur_name = "images"

    if input_format.startswith("bgr"):
        perm_t = f"{prefix}bgr_perm"
        pre_inits.append(numpy_helper.from_array(np.array([2, 1, 0], dtype=np.int64), name=perm_t))
        next_name = target_name if input_format == "bgr01" else f"{prefix}bgr_01"
        pre_nodes.append(helper.make_node("Gather", [cur_name, perm_t], [next_name], axis=1))
        cur_name = next_name

    if input_format.endswith("255"):
        scale_t = f"{prefix}scale_255"
        pre_inits.append(numpy_helper.from_array(np.array(255.0, dtype=np.float32), name=scale_t))
        pre_nodes.append(helper.make_node("Mul", [cur_name, scale_t], [target_name]))

    existing_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(pre_nodes)
    graph.node.extend(existing_nodes)
    graph.initializer.extend(pre_inits)


def _has_tail_exp_node(model: onnx.ModelProto, lookback: int = 35) -> bool:
    return any(n.op_type == "Exp" for n in model.graph.node[-lookback:])


def _build_yolox_grids_strides(
    in_h: int, in_w: int, strides: tuple[int, ...] = (8, 16, 32)
) -> tuple[np.ndarray, np.ndarray]:
    grids_list: list[np.ndarray] = []
    strides_list: list[np.ndarray] = []
    for stride in strides:
        gh, gw = in_h // stride, in_w // stride
        xv, yv = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32), indexing="xy")
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids_list.append(grid)
        strides_list.append(np.full((1, grid.shape[1], 1), float(stride), dtype=np.float32))
    return np.concatenate(grids_list, axis=1), np.concatenate(strides_list, axis=1)


def _infer_3d_dims(model: onnx.ModelProto, imgsz: int) -> tuple[int, int, int, int]:
    """Return (in_h, in_w, dim1, dim2) from static graph metadata or ORT probe."""
    inp = model.graph.input[0]
    in_dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    in_h = int(in_dims[2]) if len(in_dims) >= 4 and in_dims[2] > 0 else int(imgsz)
    in_w = int(in_dims[3]) if len(in_dims) >= 4 and in_dims[3] > 0 else int(imgsz)

    out = model.graph.output[0]
    out_dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    d1 = int(out_dims[1]) if len(out_dims) >= 3 and out_dims[1] > 0 else 0
    d2 = int(out_dims[2]) if len(out_dims) >= 3 and out_dims[2] > 0 else 0

    if d1 <= 0 or d2 <= 0:
        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 3, in_h, in_w), dtype=np.float32)
        probe = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
        if probe.ndim != 3:
            raise ValueError(f"Expected 3D raw detector output, got shape {probe.shape}")
        d1, d2 = int(probe.shape[1]), int(probe.shape[2])

    return in_h, in_w, d1, d2


def _finalize_and_save(
    model: onnx.ModelProto,
    output_path: str | Path,
    *,
    in_h: int,
    in_w: int,
    k_val: int,
    opset: int,
    has_angle: bool,
    coco_remap: bool,
    quantize: str,
) -> Path:
    final_out = "detections"
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(final_out, TensorProto.FLOAT, [1, k_val, 7])
    )
    if model.ir_version > 10:
        model.ir_version = 10

    onnx.checker.check_model(model)

    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if quantize == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
            fp32_tmp = Path(tf.name)
        try:
            onnx.save(model, str(fp32_tmp))
            quantize_dynamic(
                model_input=str(fp32_tmp),
                model_output=str(out_path),
                weight_type=QuantType.QUInt8,
            )
        finally:
            if fp32_tmp.exists():
                fp32_tmp.unlink()
    elif quantize == "none":
        onnx.save(model, str(out_path))
    else:
        raise ValueError(f"Unsupported --quantize={quantize!r}; choose 'none' or 'int8'.")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 3, in_h, in_w), dtype=np.float32)
    out_arr = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
    if out_arr.shape != (1, k_val, 7):
        raise RuntimeError(f"Wrapped ONNX output shape mismatch: {out_arr.shape} != {(1, k_val, 7)}")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[+] Exported WOG ONNX: {out_path}")
    print(f"    Input:   {sess.get_inputs()[0].name} [1, 3, {in_h}, {in_w}] float32 RGB [0, 1]")
    print(f"    Output:  {final_out} [1, {k_val}, 7] float32 [cx, cy, w, h, theta_rad, conf, cls]")
    print(f"    Mode:    obb={has_angle}, coco_remap={coco_remap}, quantize={quantize}")
    print(f"    Size:    {size_mb:.2f} MB (limit <= 40.0 MB), IR={model.ir_version}, opset={opset}")
    return out_path


def _make_graph_helpers(opset: int, prefix: str = "__wog_wrap_"):
    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    def add_init(name: str, arr: np.ndarray) -> str:
        full = prefix + name
        inits.append(numpy_helper.from_array(arr, name=full))
        return full

    def make_slice(data: str, start: int, end: int, axis: int, out_name: str) -> str:
        s_t = add_init(f"{out_name}_starts", np.array([start], dtype=np.int64))
        e_t = add_init(f"{out_name}_ends", np.array([end], dtype=np.int64))
        a_t = add_init(f"{out_name}_axes", np.array([axis], dtype=np.int64))
        dst = prefix + out_name
        nodes.append(helper.make_node("Slice", [data, s_t, e_t, a_t], [dst]))
        return dst

    def make_reduce_max(data: str, axis: int, out_name: str) -> str:
        dst = prefix + out_name
        if opset >= 18:
            ax_t = add_init(f"{out_name}_axes", np.array([axis], dtype=np.int64))
            nodes.append(helper.make_node("ReduceMax", [data, ax_t], [dst], keepdims=1))
        else:
            nodes.append(helper.make_node("ReduceMax", [data], [dst], axes=[axis], keepdims=1))
        return dst

    def make_squeeze(data: str, axis: int, out_name: str) -> str:
        dst = prefix + out_name
        if opset >= 13:
            ax_t = add_init(f"{out_name}_axes", np.array([axis], dtype=np.int64))
            nodes.append(helper.make_node("Squeeze", [data, ax_t], [dst]))
        else:
            nodes.append(helper.make_node("Squeeze", [data], [dst], axes=[axis]))
        return dst

    return inits, nodes, add_init, make_slice, make_reduce_max, make_squeeze


def _append_wog_tail(
    *,
    boxes_norm: str,
    scores_all: str,
    theta_tensor: str | None,
    n_cls: int,
    coco_remap: bool,
    k_val: int,
    prefix: str,
    nodes: list[onnx.NodeProto],
    add_init,
    make_reduce_max,
    make_squeeze,
) -> None:
    if coco_remap:
        idx_p = add_init("coco_person_idx", np.array(COCO_PERSON_INDICES, dtype=np.int64))
        idx_v = add_init("coco_vehicle_idx", np.array(COCO_VEHICLE_INDICES, dtype=np.int64))
        sc_p = prefix + "score_person"
        sc_v_all = prefix + "score_vehicle_all"
        nodes.append(helper.make_node("Gather", [scores_all, idx_p], [sc_p], axis=2))
        nodes.append(helper.make_node("Gather", [scores_all, idx_v], [sc_v_all], axis=2))
        sc_v = make_reduce_max(sc_v_all, 2, "score_vehicle")
        scores_2c = prefix + "scores_2c"
        nodes.append(helper.make_node("Concat", [sc_p, sc_v], [scores_2c], axis=2))
    else:
        if n_cls != 2:
            raise ValueError(
                f"Expected 2 classes (0=person, 1=vehicle) without --coco-remap, got num_classes={n_cls}."
            )
        scores_2c = scores_all

    conf = make_reduce_max(scores_2c, 2, "conf")
    cls_i64 = prefix + "cls_i64"
    cls_f = prefix + "cls_f"
    nodes.append(helper.make_node("ArgMax", [scores_2c], [cls_i64], axis=2, keepdims=1))
    nodes.append(helper.make_node("Cast", [cls_i64], [cls_f], to=TensorProto.FLOAT))

    if theta_tensor is None:
        zero_s = add_init("zero_scalar", np.array(0.0, dtype=np.float32))
        theta = prefix + "theta_zero"
        nodes.append(helper.make_node("Mul", [conf, zero_s], [theta]))
    else:
        theta = theta_tensor

    det_all = prefix + "det_all"
    nodes.append(helper.make_node("Concat", [boxes_norm, theta, conf, cls_f], [det_all], axis=2))

    conf_2d = make_squeeze(conf, 2, "conf_2d")
    k_t = add_init("topk_k", np.array([k_val], dtype=np.int64))
    topk_vals = prefix + "topk_vals"
    topk_idx = prefix + "topk_idx"
    nodes.append(helper.make_node("TopK", [conf_2d, k_t], [topk_vals, topk_idx], axis=1))
    topk_idx_1d = make_squeeze(topk_idx, 0, "topk_idx_1d")
    nodes.append(helper.make_node("Gather", [det_all, topk_idx_1d], ["detections"], axis=1))


def wrap_yolo_onnx(
    raw_onnx_path: str | Path,
    output_path: str | Path,
    *,
    imgsz: int = 320,
    num_classes: int | None = None,
    coco_remap: bool = False,
    obb: bool | None = None,
    top_k: int = 200,
    already_normalized: bool = False,
    quantize: str = "none",
    input_format: str = "rgb01",
) -> Path:
    """Append an in-graph WOG [1, K, 7] head to a channel-first [1, 4+C, N] or [1, 5+C, N] ONNX."""
    model = onnx.load(str(raw_onnx_path))
    opset = _get_main_opset(model)
    in_h, in_w, raw_c, raw_n = _infer_3d_dims(model, imgsz)

    if coco_remap:
        n_cls = 80
    elif num_classes is not None:
        n_cls = int(num_classes)
    else:
        if raw_c in (6, 7):
            n_cls = 2
        elif raw_c in (84, 85):
            n_cls = 80
            coco_remap = True
        else:
            raise ValueError(
                f"Cannot auto-infer num_classes from raw output channels C={raw_c}. "
                "Pass --num-classes N (and --obb if the head includes an angle channel)."
            )

    if obb is None:
        if raw_c == 4 + n_cls:
            has_angle = False
        elif raw_c == 5 + n_cls:
            has_angle = True
        else:
            raise ValueError(
                f"Raw channels C={raw_c} does not match 4+C ({4 + n_cls}) or 5+C ({5 + n_cls}) "
                f"for num_classes={n_cls}."
            )
    else:
        has_angle = bool(obb)
        expected_c = (5 if has_angle else 4) + n_cls
        if raw_c != expected_c:
            raise ValueError(
                f"Raw channels C={raw_c} does not match expected {expected_c} "
                f"(num_classes={n_cls}, obb={has_angle})."
            )

    k_val = max(1, min(int(top_k), raw_n))
    raw_out_name = model.graph.output[0].name
    p = "__wog_wrap_"
    _prepend_input_preprocess(model, input_format, p)
    inits, nodes, add_init, make_slice, make_reduce_max, make_squeeze = _make_graph_helpers(opset, p)

    raw_t = p + "raw_t"
    nodes.append(helper.make_node("Transpose", [raw_out_name], [raw_t], perm=[0, 2, 1]))

    boxes_raw = make_slice(raw_t, 0, 4, 2, "boxes_raw")
    if already_normalized:
        boxes_norm = boxes_raw
    else:
        scale = add_init("wh_scale", np.array([float(in_w), float(in_h), float(in_w), float(in_h)], dtype=np.float32))
        boxes_norm = p + "boxes_norm"
        nodes.append(helper.make_node("Div", [boxes_raw, scale], [boxes_norm]))

    cls_all = make_slice(raw_t, 4, 4 + n_cls, 2, "cls_all")
    theta = make_slice(raw_t, 4 + n_cls, 5 + n_cls, 2, "theta_rad") if has_angle else None

    _append_wog_tail(
        boxes_norm=boxes_norm,
        scores_all=cls_all,
        theta_tensor=theta,
        n_cls=n_cls,
        coco_remap=coco_remap,
        k_val=k_val,
        prefix=p,
        nodes=nodes,
        add_init=add_init,
        make_reduce_max=make_reduce_max,
        make_squeeze=make_squeeze,
    )

    model.graph.initializer.extend(inits)
    model.graph.node.extend(nodes)
    return _finalize_and_save(
        model,
        output_path,
        in_h=in_h,
        in_w=in_w,
        k_val=k_val,
        opset=opset,
        has_angle=has_angle,
        coco_remap=coco_remap,
        quantize=quantize,
    )


def wrap_yolox_onnx(
    raw_onnx_path: str | Path,
    output_path: str | Path,
    *,
    imgsz: int = 416,
    num_classes: int | None = None,
    coco_remap: bool = False,
    top_k: int = 200,
    already_normalized: bool = False,
    quantize: str = "none",
    input_format: str = "bgr255",
    decode_yolox: str | bool = "auto",
) -> Path:
    """Append an in-graph WOG [1, K, 7] head to an Apache-2.0 YOLOX [1, N, 5+C] ONNX."""
    model = onnx.load(str(raw_onnx_path))
    opset = _get_main_opset(model)
    in_h, in_w, raw_n, raw_c = _infer_3d_dims(model, imgsz)

    if coco_remap:
        n_cls = 80
    elif num_classes is not None:
        n_cls = int(num_classes)
    else:
        if raw_c == 7:
            n_cls = 2
        elif raw_c == 85:
            n_cls = 80
            coco_remap = True
        else:
            raise ValueError(
                f"Cannot auto-infer YOLOX num_classes from last dim C={raw_c}. "
                "Pass --num-classes N (expected last dim = 5 + num_classes)."
            )

    if raw_c != 5 + n_cls:
        raise ValueError(f"Expected YOLOX last dim 5+C={5 + n_cls}, got {raw_c}.")

    k_val = max(1, min(int(top_k), raw_n))
    raw_out_name = model.graph.output[0].name
    p = "__wog_yolox_"
    _prepend_input_preprocess(model, input_format, p)
    inits, nodes, add_init, make_slice, make_reduce_max, make_squeeze = _make_graph_helpers(opset, p)

    expected_n = sum((in_h // s) * (in_w // s) for s in (8, 16, 32)) if (in_h % 32 == 0 and in_w % 32 == 0) else -1
    if isinstance(decode_yolox, bool):
        if decode_yolox and raw_n != expected_n:
            raise ValueError(f"--decode-yolox requires N={expected_n} anchors for {in_w}x{in_h}, got {raw_n}.")
        do_decode = decode_yolox
    else:
        mode_dec = str(decode_yolox).lower()
        if mode_dec == "true":
            if raw_n != expected_n:
                raise ValueError(f"--decode-yolox requires N={expected_n} anchors for {in_w}x{in_h}, got {raw_n}.")
            do_decode = True
        elif mode_dec == "false":
            do_decode = False
        else:
            do_decode = raw_n == expected_n and not _has_tail_exp_node(model)

    if do_decode:
        grids_arr, strides_arr = _build_yolox_grids_strides(in_h, in_w)
        grids_t = add_init("grids", grids_arr)
        strides_t = add_init("strides", strides_arr)
        half_t = add_init("half", np.array(0.5, dtype=np.float32))
        zero_t = add_init("zero_f", np.array(0.0, dtype=np.float32))
        max_w_t = add_init("max_w", np.array(float(in_w), dtype=np.float32))
        max_h_t = add_init("max_h", np.array(float(in_h), dtype=np.float32))

        raw_xy = make_slice(raw_out_name, 0, 2, 2, "raw_xy")
        raw_wh = make_slice(raw_out_name, 2, 4, 2, "raw_wh")

        xy_g = p + "xy_g"
        cxy = p + "cxy"
        wh_e = p + "wh_e"
        wh_px = p + "wh_px"
        half_wh = p + "half_wh"
        x1y1 = p + "x1y1"
        x2y2 = p + "x2y2"
        nodes.append(helper.make_node("Add", [raw_xy, grids_t], [xy_g]))
        nodes.append(helper.make_node("Mul", [xy_g, strides_t], [cxy]))
        nodes.append(helper.make_node("Exp", [raw_wh], [wh_e]))
        nodes.append(helper.make_node("Mul", [wh_e, strides_t], [wh_px]))
        nodes.append(helper.make_node("Mul", [wh_px, half_t], [half_wh]))
        nodes.append(helper.make_node("Sub", [cxy, half_wh], [x1y1]))
        nodes.append(helper.make_node("Add", [cxy, half_wh], [x2y2]))

        x1_raw = make_slice(x1y1, 0, 1, 2, "x1_raw")
        y1_raw = make_slice(x1y1, 1, 2, 2, "y1_raw")
        x2_raw = make_slice(x2y2, 0, 1, 2, "x2_raw")
        y2_raw = make_slice(x2y2, 1, 2, 2, "y2_raw")

        def clamp_0_max(src: str, max_t: str, tag: str) -> str:
            hi_floored = p + tag + "_ge0"
            clamped = p + tag
            nodes.append(helper.make_node("Max", [src, zero_t], [hi_floored]))
            nodes.append(helper.make_node("Min", [hi_floored, max_t], [clamped]))
            return clamped

        x1 = clamp_0_max(x1_raw, max_w_t, "x1")
        x2 = clamp_0_max(x2_raw, max_w_t, "x2")
        y1 = clamp_0_max(y1_raw, max_h_t, "y1")
        y2 = clamp_0_max(y2_raw, max_h_t, "y2")

        xsum, cx_px, cx = p + "xsum", p + "cx_px", p + "cx"
        ysum, cy_px, cy = p + "ysum", p + "cy_px", p + "cy"
        bw_px, bw = p + "bw_px", p + "bw"
        bh_px, bh = p + "bh_px", p + "bh"
        nodes.append(helper.make_node("Add", [x1, x2], [xsum]))
        nodes.append(helper.make_node("Mul", [xsum, half_t], [cx_px]))
        nodes.append(helper.make_node("Div", [cx_px, max_w_t], [cx]))
        nodes.append(helper.make_node("Add", [y1, y2], [ysum]))
        nodes.append(helper.make_node("Mul", [ysum, half_t], [cy_px]))
        nodes.append(helper.make_node("Div", [cy_px, max_h_t], [cy]))
        nodes.append(helper.make_node("Sub", [x2, x1], [bw_px]))
        nodes.append(helper.make_node("Div", [bw_px, max_w_t], [bw]))
        nodes.append(helper.make_node("Sub", [y2, y1], [bh_px]))
        nodes.append(helper.make_node("Div", [bh_px, max_h_t], [bh]))

        boxes_norm = p + "boxes_norm"
        nodes.append(helper.make_node("Concat", [cx, cy, bw, bh], [boxes_norm], axis=2))
        print(f"    YOLOX decode: baked RGB[0,1]->{input_format} + [8,16,32] grid decode ({raw_n} anchors)")
    else:
        boxes_raw = make_slice(raw_out_name, 0, 4, 2, "boxes_raw")
        if already_normalized:
            boxes_norm = boxes_raw
        else:
            scale = add_init("wh_scale", np.array([float(in_w), float(in_h), float(in_w), float(in_h)], dtype=np.float32))
            boxes_norm = p + "boxes_norm"
            nodes.append(helper.make_node("Div", [boxes_raw, scale], [boxes_norm]))
        print(f"    YOLOX preprocess: baked RGB[0,1]->{input_format} (grid decode skipped; N={raw_n})")

    # YOLOX final class confidence = obj_conf * cls_prob
    obj_conf = make_slice(raw_out_name, 4, 5, 2, "obj_conf")
    cls_raw = make_slice(raw_out_name, 5, 5 + n_cls, 2, "cls_raw")
    scores_all = p + "scores_all"
    nodes.append(helper.make_node("Mul", [obj_conf, cls_raw], [scores_all]))

    _append_wog_tail(
        boxes_norm=boxes_norm,
        scores_all=scores_all,
        theta_tensor=None,
        n_cls=n_cls,
        coco_remap=coco_remap,
        k_val=k_val,
        prefix=p,
        nodes=nodes,
        add_init=add_init,
        make_reduce_max=make_reduce_max,
        make_squeeze=make_squeeze,
    )

    model.graph.initializer.extend(inits)
    model.graph.node.extend(nodes)
    return _finalize_and_save(
        model,
        output_path,
        in_h=in_h,
        in_w=in_w,
        k_val=k_val,
        opset=opset,
        has_angle=False,
        coco_remap=coco_remap,
        quantize=quantize,
    )


def _update_manifest(manifest_path: Path, model_rel: str, imgsz: int) -> None:
    if not manifest_path.is_file():
        print(f"[!] Skipping --update-manifest: {manifest_path} not found.")
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for stage in data.get("stages", []):
        if isinstance(stage, dict) and stage.get("type") == "detector":
            stage["model"] = model_rel.replace("\\", "/")
            stage["input_size"] = [int(imgsz), int(imgsz)]
            break
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Updated {manifest_path.name}: model={model_rel}, input_size=[{imgsz}, {imgsz}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wrap a permissive ONNX detector into the WOG [1, K, 7] contract.")
    ap.add_argument(
        "--mode",
        choices=("wrap-yolox", "wrap-onnx"),
        required=True,
        help="wrap-yolox: [1, N, 5+C] Apache-2.0 YOLOX head; wrap-onnx: [1, 4+C, N] AABB or [1, 5+C, N] OBB head",
    )
    ap.add_argument("--onnx", required=True, help="Path to existing raw .onnx export")
    ap.add_argument("--imgsz", type=int, default=320, help="Square input size (32-512; recommended 320 or 416)")
    ap.add_argument("--num-classes", type=int, default=None, help="Number of classes in raw head (2 or 80)")
    ap.add_argument("--coco-remap", action="store_true", help="Remap 80-class COCO head -> 0=person, 1=vehicle")
    ap.add_argument("--obb", action="store_true", default=None, help="Force OBB head for --mode wrap-onnx (5+C channels)")
    ap.add_argument("--top-k", type=int, default=200, help="Top-K detections kept in-graph (default 200)")
    ap.add_argument("--already-normalized", action="store_true", help="Set if raw boxes are already in [0, 1]")
    ap.add_argument("--quantize", choices=("none", "int8"), default="none", help="Optional dynamic INT8 quantization")
    ap.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        default=None,
        help="Inner ONNX expected input format before wrapping (default: bgr255 for wrap-yolox, rgb01 for wrap-onnx)",
    )
    dec_grp = ap.add_mutually_exclusive_group()
    dec_grp.add_argument("--decode-yolox", dest="decode_yolox", action="store_const", const="true",
                         help="Force YOLOX [8,16,32] grid decoding inside ONNX")
    dec_grp.add_argument("--no-decode-yolox", dest="decode_yolox", action="store_const", const="false",
                         help="Disable YOLOX grid decoding (when ONNX already exported decoded)")
    ap.set_defaults(decode_yolox="auto")
    ap.add_argument("--out", default="models/detector.onnx", help="Output .onnx path inside repo")
    ap.add_argument("--update-manifest", action="store_true", help="Update model.json detector path & input_size")
    args = ap.parse_args(argv)

    if not 32 <= args.imgsz <= 512:
        raise ValueError(f"--imgsz must be 32..512 for la-frite-1gb, got {args.imgsz}")

    raw_path = Path(args.onnx).resolve()
    if args.mode == "wrap-yolox":
        fmt = args.input_format or "bgr255"
        out_path = wrap_yolox_onnx(
            raw_path,
            args.out,
            imgsz=args.imgsz,
            num_classes=args.num_classes,
            coco_remap=args.coco_remap,
            top_k=args.top_k,
            already_normalized=args.already_normalized,
            quantize=args.quantize,
            input_format=fmt,
            decode_yolox=args.decode_yolox,
        )
    else:
        fmt = args.input_format or "rgb01"
        out_path = wrap_yolo_onnx(
            raw_path,
            args.out,
            imgsz=args.imgsz,
            num_classes=args.num_classes,
            coco_remap=args.coco_remap,
            obb=True if args.obb else None,
            top_k=args.top_k,
            already_normalized=args.already_normalized,
            quantize=args.quantize,
            input_format=fmt,
        )

    if args.update_manifest:
        repo_root = Path(__file__).resolve().parent
        try:
            rel = out_path.relative_to(repo_root).as_posix()
        except ValueError:
            rel = str(args.out).replace("\\", "/")
        _update_manifest(repo_root / "model.json", rel, args.imgsz)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
