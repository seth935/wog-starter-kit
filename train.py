"""Weights of Giants — Season Zero Permissive Dataset Prep Helper (wog-cv-v2.1).

LICENSE POLICY (Permissive Open-Source Only):
  Season Zero accepts permissive SPDX licenses ONLY (Apache-2.0, MIT, BSD-2/3-Clause,
  ISC, 0BSD, Unlicense, CC0-1.0, Zlib, BSL-1.0, CC-BY-4.0).
  Copyleft licenses (AGPL-3.0, GPL, LGPL, MPL-2.0, EPL-2.0, CC-BY-SA) — including
  Ultralytics YOLO11 / YOLOv8 checkpoints — are NOT accepted.

  Recommended permissive starting detectors:
    - YOLOX-Nano / YOLOX-Tiny (Apache-2.0): https://github.com/Megvii-BaseDetection/YOLOX
    - NanoDet-Plus (Apache-2.0):            https://github.com/RangiLyu/nanodet
    - PP-PicoDet-S (Apache-2.0):            https://github.com/PaddlePaddle/PaddleDetection
    - TorchVision SSDLite (BSD-3-Clause):   https://pytorch.org/vision

Subcommands:
  1. prepare-coco-json  Filter a standard COCO instances_*.json into a 2-class WOG COCO JSON
                        (1=person, 2=vehicle) ready for YOLOX, NanoDet-Plus, PP-PicoDet, or TorchVision.
  2. prepare-coco       Remap 80-class YOLO-format .txt labels -> 2-class WOG labels
                        (0=person, 1=vehicle: bicycle/car/motorcycle/bus/truck) and write data.yaml.

Examples:
    # 1. Prepare 2-class COCO JSON for Apache-2.0 YOLOX / NanoDet-Plus / PP-PicoDet
    python train.py prepare-coco-json \
      --src-json datasets/coco/annotations/instances_train2017.json \
      --dst-json datasets/wog_2class/instances_train.json

    # 2. Remap YOLO-format .txt labels to 2 WOG classes
    python train.py prepare-coco \
      --src-labels datasets/coco/labels \
      --dst-labels datasets/wog_2class/labels \
      --images-dir datasets/coco/images \
      --out-yaml datasets/wog_2class/data.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# COCO 80-class 0-based IDs -> WOG Season Zero 0-based classes:
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

# Standard COCO JSON category_id (1-based) -> WOG 2-class COCO JSON category_id:
#   1 = person (COCO id 1)
#   2 = vehicle (COCO ids: bicycle=2, car=3, motorcycle=4, bus=6, truck=8)
COCO_JSON_CAT_TO_WOG: dict[int, int] = {
    1: 1,  # person
    2: 2,  # bicycle -> vehicle
    3: 2,  # car -> vehicle
    4: 2,  # motorcycle -> vehicle
    6: 2,  # bus -> vehicle
    8: 2,  # truck -> vehicle
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


def prepare_coco_json(
    src_json_path: str | Path,
    dst_json_path: str | Path,
    *,
    keep_empty_images: bool = False,
) -> dict[str, int]:
    """Filter & remap a standard COCO instances JSON to WOG 2-class (1=person, 2=vehicle).

    Output JSON is directly compatible with Apache-2.0 YOLOX, NanoDet-Plus, PP-PicoDet,
    and BSD-3-Clause TorchVision dataset loaders.
    """
    src = Path(src_json_path).resolve()
    dst = Path(dst_json_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Source COCO JSON not found: {src}")

    data = json.loads(src.read_text(encoding="utf-8"))
    new_anns: list[dict] = []
    used_image_ids: set[int] = set()
    n_person = 0
    n_vehicle = 0

    for ann in data.get("annotations", []):
        cat_id = int(ann.get("category_id", -1))
        if cat_id not in COCO_JSON_CAT_TO_WOG:
            continue
        wog_cat = COCO_JSON_CAT_TO_WOG[cat_id]
        copied = dict(ann)
        copied["category_id"] = wog_cat
        new_anns.append(copied)
        used_image_ids.add(int(ann["image_id"]))
        if wog_cat == 1:
            n_person += 1
        else:
            n_vehicle += 1

    all_images = data.get("images", [])
    new_images = (
        all_images
        if keep_empty_images
        else [img for img in all_images if int(img["id"]) in used_image_ids]
    )

    out_payload = {
        "info": data.get("info", {"description": "WOG Season Zero 2-class COCO dataset"}),
        "licenses": data.get("licenses", []),
        "images": new_images,
        "annotations": new_anns,
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "vehicle", "supercategory": "vehicle"},
        ],
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out_payload), encoding="utf-8")
    print(
        f"[+] Wrote 2-class COCO JSON -> {dst} "
        f"(images={len(new_images)}, person={n_person}, vehicle={n_vehicle})"
    )
    return {"images": len(new_images), "person": n_person, "vehicle": n_vehicle}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Weights of Giants Season Zero permissive dataset helper.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_json = sub.add_parser(
        "prepare-coco-json",
        help="Remap COCO instances JSON -> 2-class COCO JSON (for YOLOX, NanoDet-Plus, PP-PicoDet, TorchVision)",
    )
    p_json.add_argument("--src-json", required=True, help="Source COCO instances_*.json")
    p_json.add_argument("--dst-json", required=True, help="Destination 2-class COCO JSON")
    p_json.add_argument("--keep-empty-images", action="store_true", help="Keep images with zero person/vehicle boxes")

    p_prep = sub.add_parser("prepare-coco", help="Remap COCO-80 YOLO .txt labels to WOG 2-class (person/vehicle)")
    p_prep.add_argument("--src-labels", required=True, help="Source YOLO labels directory")
    p_prep.add_argument("--dst-labels", required=True, help="Destination 2-class labels directory")
    p_prep.add_argument("--images-dir", default=None, help="Optional images root for data.yaml")
    p_prep.add_argument("--out-yaml", default=None, help="Optional output data.yaml path")

    args = ap.parse_args(argv)
    if args.cmd == "prepare-coco-json":
        prepare_coco_json(
            args.src_json,
            args.dst_json,
            keep_empty_images=bool(args.keep_empty_images),
        )
        return 0

    prepare_coco_labels(
        args.src_labels,
        args.dst_labels,
        images_dir=args.images_dir,
        out_yaml=args.out_yaml,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
