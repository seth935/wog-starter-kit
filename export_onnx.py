"""Weights of Giants — ONNX Contract Exporter & Wrapper (wog-cv-v2.1).

LICENSE NOTICE:
  - Models exported from Ultralytics YOLO11 / YOLOv8 checkpoints are covered by AGPL-3.0
    (accepted by Season Zero). License your submission repo AGPL-3.0 — not Apache-2.0/MIT.
  - For Apache-2.0 / MIT submissions, start from Apache-2.0 architectures (YOLOX, NanoDet-Plus,
    PP-PicoDet) and wrap their ONNX outputs with `--mode wrap-onnx`.

Wraps a trained Ultralytics YOLO / YOLO-OBB checkpoint (.pt) or raw detection ONNX
([1, 4+C, N] AABB or [1, 5+C, N] OBB) into the exact La Frite ONNX Runtime contract:

    Input:  images     [1, 3, H, W]  float32, RGB in [0.0, 1.0]
    Output: detections [1, K, 7]     float32, [cx, cy, w, h, theta_rad, conf, cls]
            - cx, cy, w, h normalized to [0, 1] relative to [W, H]
            - theta_rad in radians (0.0 for upright AABB detectors)
            - conf in [0, 1]
            - cls: 0 = person, 1 = vehicle

Examples:
    # 1. Export + wrap a custom 2-class Ultralytics YOLO or YOLO-OBB checkpoint
    python export_onnx.py --mode yolo --weights work/runs/wog_exp/weights/best.pt --imgsz 320 --out models/detector.onnx --update-manifest

    # 2. Export + wrap a pretrained 80-class COCO YOLO checkpoint (auto-remaps COCO -> 0=person, 1=vehicle)
    python export_onnx.py --mode yolo --weights work/yolo11n.pt --imgsz 320 --coco-remap --quantize int8 --out models/detector_int8.onnx --update-manifest

    # 3. Wrap an existing raw YOLO-format ONNX ([1, 4+C, N] or [1, 5+C, N]) without PyTorch
    python export_onnx.py --mode wrap-onnx --onnx work/raw_yolo.onnx --imgsz 320 --num-classes 2 --out models/detector.onnx
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


def _infer_dims(model: onnx.ModelProto, imgsz: int) -> tuple[int, int, int, int]:
    """Return (in_h, in_w, raw_channels, num_anchors) from static graph metadata or ORT probe."""
    inp = model.graph.input[0]
    in_dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    in_h = int(in_dims[2]) if len(in_dims) >= 4 and in_dims[2] > 0 else int(imgsz)
    in_w = int(in_dims[3]) if len(in_dims) >= 4 and in_dims[3] > 0 else int(imgsz)

    out = model.graph.output[0]
    out_dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    raw_c = int(out_dims[1]) if len(out_dims) >= 3 and out_dims[1] > 0 else 0
    raw_n = int(out_dims[2]) if len(out_dims) >= 3 and out_dims[2] > 0 else 0

    if raw_c <= 0 or raw_n <= 0:
        import onnxruntime as ort

        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 3, in_h, in_w), dtype=np.float32)
        probe = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
        if probe.ndim != 3:
            raise ValueError(f"Expected 3D raw detector output [1, C, N], got shape {probe.shape}")
        raw_c, raw_n = int(probe.shape[1]), int(probe.shape[2])

    return in_h, in_w, raw_c, raw_n


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
) -> Path:
    """Append an in-graph WOG [1, K, 7] post-processing head to a raw YOLO/YOLO-OBB ONNX."""
    model = onnx.load(str(raw_onnx_path))
    opset = _get_main_opset(model)
    in_h, in_w, raw_c, raw_n = _infer_dims(model, imgsz)

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

    inits: list[onnx.TensorProto] = []
    nodes: list[onnx.NodeProto] = []

    def add_init(name: str, arr: np.ndarray) -> str:
        full = p + name
        inits.append(numpy_helper.from_array(arr, name=full))
        return full

    def make_slice(data: str, start: int, end: int, axis: int, out_name: str) -> str:
        s_t = add_init(f"{out_name}_starts", np.array([start], dtype=np.int64))
        e_t = add_init(f"{out_name}_ends", np.array([end], dtype=np.int64))
        a_t = add_init(f"{out_name}_axes", np.array([axis], dtype=np.int64))
        dst = p + out_name
        nodes.append(helper.make_node("Slice", [data, s_t, e_t, a_t], [dst]))
        return dst

    def make_reduce_max(data: str, axis: int, out_name: str) -> str:
        dst = p + out_name
        if opset >= 18:
            ax_t = add_init(f"{out_name}_axes", np.array([axis], dtype=np.int64))
            nodes.append(helper.make_node("ReduceMax", [data, ax_t], [dst], keepdims=1))
        else:
            nodes.append(helper.make_node("ReduceMax", [data], [dst], axes=[axis], keepdims=1))
        return dst

    def make_squeeze(data: str, axis: int, out_name: str) -> str:
        dst = p + out_name
        if opset >= 13:
            ax_t = add_init(f"{out_name}_axes", np.array([axis], dtype=np.int64))
            nodes.append(helper.make_node("Squeeze", [data, ax_t], [dst]))
        else:
            nodes.append(helper.make_node("Squeeze", [data], [dst], axes=[axis]))
        return dst

    # 1. [1, C_total, N] -> [1, N, C_total]
    raw_t = p + "raw_t"
    nodes.append(helper.make_node("Transpose", [raw_out_name], [raw_t], perm=[0, 2, 1]))

    # 2. Boxes [1, N, 4] normalized to [0, 1] relative to [in_w, in_h]
    boxes_raw = make_slice(raw_t, 0, 4, 2, "boxes_raw")
    if already_normalized:
        boxes_norm = boxes_raw
    else:
        scale = add_init("wh_scale", np.array([float(in_w), float(in_h), float(in_w), float(in_h)], dtype=np.float32))
        boxes_norm = p + "boxes_norm"
        nodes.append(helper.make_node("Div", [boxes_raw, scale], [boxes_norm]))

    # 3. Class scores -> 2-class [person, vehicle]
    cls_all = make_slice(raw_t, 4, 4 + n_cls, 2, "cls_all")
    if coco_remap:
        idx_p = add_init("coco_person_idx", np.array(COCO_PERSON_INDICES, dtype=np.int64))
        idx_v = add_init("coco_vehicle_idx", np.array(COCO_VEHICLE_INDICES, dtype=np.int64))
        sc_p = p + "score_person"
        sc_v_all = p + "score_vehicle_all"
        nodes.append(helper.make_node("Gather", [cls_all, idx_p], [sc_p], axis=2))
        nodes.append(helper.make_node("Gather", [cls_all, idx_v], [sc_v_all], axis=2))
        sc_v = make_reduce_max(sc_v_all, 2, "score_vehicle")
        scores_2c = p + "scores_2c"
        nodes.append(helper.make_node("Concat", [sc_p, sc_v], [scores_2c], axis=2))
    else:
        if n_cls != 2:
            raise ValueError(
                f"Expected 2 classes (0=person, 1=vehicle) without --coco-remap, got num_classes={n_cls}."
            )
        scores_2c = cls_all

    conf = make_reduce_max(scores_2c, 2, "conf")
    cls_i64 = p + "cls_i64"
    cls_f = p + "cls_f"
    nodes.append(helper.make_node("ArgMax", [scores_2c], [cls_i64], axis=2, keepdims=1))
    nodes.append(helper.make_node("Cast", [cls_i64], [cls_f], to=TensorProto.FLOAT))

    # 4. Angle theta_rad [1, N, 1]
    if has_angle:
        theta = make_slice(raw_t, 4 + n_cls, 5 + n_cls, 2, "theta_rad")
    else:
        zero_s = add_init("zero_scalar", np.array(0.0, dtype=np.float32))
        theta = p + "theta_zero"
        nodes.append(helper.make_node("Mul", [conf, zero_s], [theta]))

    # 5. Concat -> [1, N, 7] = [cx, cy, w, h, theta_rad, conf, cls]
    det_all = p + "det_all"
    nodes.append(helper.make_node("Concat", [boxes_norm, theta, conf, cls_f], [det_all], axis=2))

    # 6. Top-K by confidence -> [1, K, 7]
    conf_2d = make_squeeze(conf, 2, "conf_2d")
    k_t = add_init("topk_k", np.array([k_val], dtype=np.int64))
    topk_vals = p + "topk_vals"
    topk_idx = p + "topk_idx"
    nodes.append(helper.make_node("TopK", [conf_2d, k_t], [topk_vals, topk_idx], axis=1))
    topk_idx_1d = make_squeeze(topk_idx, 0, "topk_idx_1d")
    final_out = "detections"
    nodes.append(helper.make_node("Gather", [det_all, topk_idx_1d], [final_out], axis=1))

    model.graph.initializer.extend(inits)
    model.graph.node.extend(nodes)

    # Replace graph outputs with single WOG contract output
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

    # Smoke-test the saved ONNX via ONNX Runtime
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


def export_ultralytics_pt(weights_path: str | Path, imgsz: int) -> Path:
    """Export an Ultralytics YOLO / YOLO-OBB .pt checkpoint to a temporary raw ONNX."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for --mode yolo. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(str(weights_path))
    exported = model.export(
        format="onnx",
        imgsz=int(imgsz),
        opset=17,
        simplify=False,
        dynamic=False,
        nms=False,
    )
    return Path(exported).resolve()


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
    ap = argparse.ArgumentParser(description="Export and wrap a detector into the WOG [1, K, 7] ONNX contract.")
    ap.add_argument("--mode", choices=("yolo", "wrap-onnx"), required=True)
    ap.add_argument("--weights", help="Path to Ultralytics .pt checkpoint (for --mode yolo)")
    ap.add_argument("--onnx", help="Path to existing raw YOLO-format .onnx (for --mode wrap-onnx)")
    ap.add_argument("--imgsz", type=int, default=320, help="Square input size (32-512; recommended 320 or 416)")
    ap.add_argument("--num-classes", type=int, default=None, help="Number of classes in raw head (2 or 80)")
    ap.add_argument("--coco-remap", action="store_true", help="Remap 80-class COCO head -> 0=person, 1=vehicle")
    ap.add_argument("--obb", action="store_true", default=None, help="Force OBB head (raw has 5+C channels)")
    ap.add_argument("--top-k", type=int, default=200, help="Top-K detections kept in-graph (default 200)")
    ap.add_argument("--already-normalized", action="store_true", help="Set if raw boxes are already in [0, 1]")
    ap.add_argument("--quantize", choices=("none", "int8"), default="none", help="Optional dynamic INT8 quantization")
    ap.add_argument("--out", default="models/detector.onnx", help="Output .onnx path inside repo")
    ap.add_argument("--update-manifest", action="store_true", help="Update model.json detector path & input_size")
    args = ap.parse_args(argv)

    if not 32 <= args.imgsz <= 512:
        raise ValueError(f"--imgsz must be 32..512 for la-frite-1gb, got {args.imgsz}")

    tmp_raw: Path | None = None
    try:
        if args.mode == "yolo":
            if not args.weights:
                raise ValueError("--weights is required for --mode yolo")
            tmp_raw = export_ultralytics_pt(args.weights, args.imgsz)
            raw_path = tmp_raw
        else:
            if not args.onnx:
                raise ValueError("--onnx is required for --mode wrap-onnx")
            raw_path = Path(args.onnx).resolve()

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
        )
    finally:
        if tmp_raw is not None and tmp_raw.exists() and tmp_raw != Path(args.out).resolve():
            try:
                tmp_raw.unlink()
            except OSError:
                pass

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
