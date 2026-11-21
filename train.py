"""Weights of Giants — Season Zero Edge Detector Training & Dataset Helper (wog-cv-v2.1).

LICENSE NOTICE:
  - Ultralytics YOLO11 / YOLOv8 weights and fine-tuned checkpoints are covered by AGPL-3.0
    (which Season Zero accepts!). If you train or export from Ultralytics, your submission
    repository MUST be licensed AGPL-3.0 — do NOT label it MIT or Apache-2.0.
  - If you want permissive Apache-2.0 / MIT licensing, train/export from Apache-2.0
    detectors such as YOLOX-Nano, NanoDet-Plus, or PP-PicoDet-S instead.

Subcommands:
  1. prepare-coco   Remap standard 80-class COCO YOLO labels -> 2-class WOG labels
                    (0=person, 1=vehicle: bicycle/car/motorcycle/bus/truck) and write data.yaml.
  2. train          Train a lightweight edge detector (Ultralytics YOLO / YOLO-OBB) into
                    the .gitignored `work/runs/` folder so `.pt` checkpoints never pollute
                    your submission repository.

Examples:
    # Step 1: Remap an existing YOLO-format COCO dataset to 2 WOG classes
    python train.py prepare-coco --src-labels datasets/coco/labels --dst-labels datasets/wog_2class/labels --images-dir datasets/coco/images --out-yaml datasets/wog_2class/data.yaml

    # Step 2: Train a lightweight 320x320 detector (stored in work/runs/, never committed)
    python train.py train --data datasets/wog_2class/data.yaml --model yolo11n.pt --imgsz 320 --epochs 50

    # Step 3: Export & wrap to WOG [1, K, 7] ONNX contract
    python export_onnx.py --mode yolo --weights work/runs/wog_train/weights/best.pt --imgsz 320 --out models/detector.onnx --update-manifest
"""

from __future__ import annotations

import argparse
from pathlib import Path

# COCO 80-class IDs mapped to WOG Season Zero classes:
#   0 = person
#   1 = vehicle (bicycle=1, car=2, motorcycle=3, bus=5, truck=7)
COCO_TO_WOG: dict[int, int] = {
    0: 0,  # person
    1: 1,  # bicycle -> vehicle
    2: 1,  # car -> vehicle
    3: 1,  # motorcycle -> vehicle
    5: 1,  # bus -> vehicle
    7: 1,  # truck -> vehicle
}


def remap_yolo_label_lines(lines: list[str], class_map: dict[int, int] | None = None) -> list[str]:
    """Remap YOLO detection or YOLO-OBB label lines using class_map; drop unmapped classes."""
    cmap = COCO_TO_WOG if class_map is None else class_map
    out: list[str] = []
    for raw in lines:
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            src_cls = int(float(parts[0]))
        except ValueError:
            continue
        if src_cls not in cmap:
            continue
        out.append(" ".join([str(cmap[src_cls]), *parts[1:]]))
    return out


def prepare_coco_labels(
    src_labels_dir: str | Path,
    dst_labels_dir: str | Path,
    *,
    images_dir: str | Path | None = None,
    out_yaml: str | Path | None = None,
) -> dict[str, int]:
    """Remap a directory tree of YOLO .txt label files from COCO-80 to WOG 2-class."""
    src_root = Path(src_labels_dir).resolve()
    dst_root = Path(dst_labels_dir).resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"Source labels directory not found: {src_root}")

    files_written = 0
    boxes_person = 0
    boxes_vehicle = 0

    for txt_path in sorted(src_root.rglob("*.txt")):
        rel = txt_path.relative_to(src_root)
        target = dst_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        remapped = remap_yolo_label_lines(lines)
        target.write_text(("\n".join(remapped) + "\n") if remapped else "", encoding="utf-8")
        files_written += 1
        for ln in remapped:
            if ln.startswith("0 "):
                boxes_person += 1
            elif ln.startswith("1 "):
                boxes_vehicle += 1

    if out_yaml:
        yp = Path(out_yaml).resolve()
        yp.parent.mkdir(parents=True, exist_ok=True)
        img_root = Path(images_dir).resolve() if images_dir else yp.parent / "images"
        yaml_text = (
            "# Weights of Giants Season Zero 2-class dataset config (wog-cv-v2.1)\n"
            f"path: {img_root.parent.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: person\n"
            "  1: vehicle\n"
        )
        yp.write_text(yaml_text, encoding="utf-8")
        print(f"[+] Wrote dataset YAML: {yp}")

    print(
        f"[+] Remapped {files_written} label files -> {dst_root} "
        f"(person={boxes_person}, vehicle={boxes_vehicle})"
    )
    return {"files": files_written, "person": boxes_person, "vehicle": boxes_vehicle}


def run_train(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for training. Install with: pip install ultralytics torch"
        ) from exc

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not 32 <= args.imgsz <= 512:
        raise ValueError(f"--imgsz must be 32..512 for la-frite-1gb, got {args.imgsz}")

    # Store all checkpoints inside .gitignored work/ so .pt files never enter git.
    project_dir = Path(args.project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading base weights: {args.model}")
    print(f"[+] Training for {args.epochs} epochs at {args.imgsz}x{args.imgsz} (batch={args.batch})")
    print(f"[+] Checkpoints saved under .gitignored directory: {project_dir / args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        imgsz=int(args.imgsz),
        epochs=int(args.epochs),
        batch=int(args.batch),
        device=args.device,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        # Season Zero CCTV / traffic-cam robustness augmentations:
        # keep scale jitter and lighting/color shifts strong while avoiding upside-down flips.
        hsv_h=0.015,
        hsv_s=0.65,
        hsv_v=0.55,
        degrees=12.0 if args.obb else 0.0,
        scale=0.55,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.8,
        erasing=0.25,
        close_mosaic=10,
    )

    best_pt = project_dir / args.name / "weights" / "best.pt"
    obb_flag = " --obb" if args.obb else ""
    print("\n[+] Training complete!")
    print("[+] Next step — export & wrap into WOG [1, K, 7] ONNX (do NOT commit .pt files):")
    print(
        f"    python export_onnx.py --mode yolo --weights {best_pt} "
        f"--imgsz {args.imgsz}{obb_flag} --out models/detector.onnx --update-manifest"
    )
    print("    python score_local.py")
    print("    python verify_local.py")
    print("    [LICENSE] Ultralytics-derived models MUST be submitted under AGPL-3.0.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Weights of Giants Season Zero training & dataset helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare-coco", help="Remap COCO-80 YOLO labels to WOG 2-class (person/vehicle)")
    p_prep.add_argument("--src-labels", required=True, help="Source YOLO labels directory")
    p_prep.add_argument("--dst-labels", required=True, help="Destination 2-class labels directory")
    p_prep.add_argument("--images-dir", default=None, help="Optional images root for data.yaml")
    p_prep.add_argument("--out-yaml", default=None, help="Optional output data.yaml path")

    p_tr = sub.add_parser("train", help="Train a lightweight YOLO / YOLO-OBB model into work/runs/")
    p_tr.add_argument("--data", required=True, help="Path to 2-class data.yaml")
    p_tr.add_argument("--model", default="yolo11n.pt", help="Base checkpoint (e.g. yolo11n.pt, yolo11n-obb.pt)")
    p_tr.add_argument("--imgsz", type=int, default=320, help="Training/export resolution (recommended 320 or 416)")
    p_tr.add_argument("--epochs", type=int, default=50, help="Training epochs")
    p_tr.add_argument("--batch", type=int, default=32, help="Batch size")
    p_tr.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    p_tr.add_argument("--obb", action="store_true", help="Enable OBB rotation augmentation (for YOLO-OBB datasets)")
    p_tr.add_argument("--project", default="work/runs", help="Output directory (keep inside .gitignored work/)")
    p_tr.add_argument("--name", default="wog_train", help="Run folder name")

    args = ap.parse_args(argv)
    if args.cmd == "prepare-coco":
        prepare_coco_labels(
            args.src_labels,
            args.dst_labels,
            images_dir=args.images_dir,
            out_yaml=args.out_yaml,
        )
        return 0
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
