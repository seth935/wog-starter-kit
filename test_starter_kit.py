"""Unit tests for the Weights of Giants Season Zero Starter Kit."""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

STARTER_DIR = Path(__file__).resolve().parent
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

import export_onnx
import score_local
import train
import verify_local


def _build_dummy_raw_yolo_onnx(path: Path, raw_arr: np.ndarray, imgsz: int = 320, opset: int = 17) -> None:
    """Build a minimal ONNX graph that maps [1, 3, H, W] -> raw_arr ([1, C, N])."""
    c_dim, n_dim = int(raw_arr.shape[1]), int(raw_arr.shape[2])
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, c_dim, n_dim])

    conv_w = numpy_helper.from_array(np.zeros((1, 3, 1, 1), dtype=np.float32), name="w0")
    conv_b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="b0")
    raw_init = numpy_helper.from_array(raw_arr.astype(np.float32), name="raw_const")
    sh_init = numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), name="sh0")

    nodes = [
        helper.make_node("Conv", ["images", "w0", "b0"], ["c0"]),
        helper.make_node("GlobalAveragePool", ["c0"], ["p0"]),
        helper.make_node("Reshape", ["p0", "sh0"], ["g0"]),
        helper.make_node("Add", ["raw_const", "g0"], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "raw_yolo_stub", [inp], [out], initializer=[conv_w, conv_b, raw_init, sh_init])
    model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", opset)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


class TestStarterKit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="wog_starter_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_wrap_2class_aabb_onnx(self) -> None:
        raw = np.zeros((1, 6, 12), dtype=np.float32)
        # Anchor 3: person at center (160, 160), size (64, 96), conf 0.88
        raw[0, 0:4, 3] = [160.0, 160.0, 64.0, 96.0]
        raw[0, 4, 3] = 0.88
        raw[0, 5, 3] = 0.05
        # Anchor 7: vehicle at (80, 240), size (128, 64), conf 0.92
        raw[0, 0:4, 7] = [80.0, 240.0, 128.0, 64.0]
        raw[0, 4, 7] = 0.02
        raw[0, 5, 7] = 0.92

        raw_p = self.tmp / "raw_2c.onnx"
        out_p = self.tmp / "wrapped_2c.onnx"
        _build_dummy_raw_yolo_onnx(raw_p, raw, imgsz=320, opset=17)

        export_onnx.wrap_yolo_onnx(raw_p, out_p, imgsz=320, num_classes=2, top_k=5)
        sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        dets = sess.run(None, {"images": np.zeros((1, 3, 320, 320), dtype=np.float32)})[0]
        self.assertEqual(dets.shape, (1, 5, 7))
        # Top-1 is vehicle (0.92)
        np.testing.assert_allclose(dets[0, 0, :4], [0.25, 0.75, 0.4, 0.2], atol=1e-4)
        self.assertAlmostEqual(float(dets[0, 0, 4]), 0.0, places=4)
        self.assertAlmostEqual(float(dets[0, 0, 5]), 0.92, places=4)
        self.assertEqual(int(round(float(dets[0, 0, 6]))), 1)
        # Top-2 is person (0.88)
        np.testing.assert_allclose(dets[0, 1, :4], [0.5, 0.5, 0.2, 0.3], atol=1e-4)
        self.assertAlmostEqual(float(dets[0, 1, 5]), 0.88, places=4)
        self.assertEqual(int(round(float(dets[0, 1, 6]))), 0)

    def test_wrap_2class_obb_onnx_preserves_theta(self) -> None:
        raw = np.zeros((1, 7, 8), dtype=np.float32)
        raw[0, 0:4, 2] = [160.0, 160.0, 80.0, 40.0]
        raw[0, 4, 2] = 0.10
        raw[0, 5, 2] = 0.91
        raw[0, 6, 2] = 0.42  # radians (~24 deg)

        raw_p = self.tmp / "raw_obb.onnx"
        out_p = self.tmp / "wrapped_obb.onnx"
        _build_dummy_raw_yolo_onnx(raw_p, raw, imgsz=320, opset=17)

        export_onnx.wrap_yolo_onnx(raw_p, out_p, imgsz=320, num_classes=2, obb=True, top_k=4)
        sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        dets = sess.run(None, {"images": np.zeros((1, 3, 320, 320), dtype=np.float32)})[0]
        self.assertEqual(dets.shape, (1, 4, 7))
        self.assertAlmostEqual(float(dets[0, 0, 4]), 0.42, places=4)
        self.assertAlmostEqual(float(dets[0, 0, 5]), 0.91, places=4)
        self.assertEqual(int(round(float(dets[0, 0, 6]))), 1)

    def test_wrap_coco80_remap_ignores_non_target_classes(self) -> None:
        raw = np.zeros((1, 84, 10), dtype=np.float32)
        # Anchor 1: COCO class 16 (dog -> ignored!) with high score 0.99
        raw[0, 0:4, 1] = [100.0, 100.0, 50.0, 50.0]
        raw[0, 4 + 16, 1] = 0.99
        # Anchor 4: COCO class 7 (truck -> vehicle=1) with score 0.77
        raw[0, 0:4, 4] = [200.0, 160.0, 100.0, 60.0]
        raw[0, 4 + 7, 4] = 0.77
        # Anchor 6: COCO class 0 (person -> person=0) with score 0.65
        raw[0, 0:4, 6] = [120.0, 200.0, 40.0, 90.0]
        raw[0, 4 + 0, 6] = 0.65

        raw_p = self.tmp / "raw_coco.onnx"
        out_p = self.tmp / "wrapped_coco.onnx"
        _build_dummy_raw_yolo_onnx(raw_p, raw, imgsz=320, opset=17)

        export_onnx.wrap_yolo_onnx(raw_p, out_p, imgsz=320, coco_remap=True, top_k=4)
        sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        dets = sess.run(None, {"images": np.zeros((1, 3, 320, 320), dtype=np.float32)})[0]
        self.assertAlmostEqual(float(dets[0, 0, 5]), 0.77, places=4)
        self.assertEqual(int(round(float(dets[0, 0, 6]))), 1)
        self.assertAlmostEqual(float(dets[0, 1, 5]), 0.65, places=4)
        self.assertEqual(int(round(float(dets[0, 1, 6]))), 0)
        # Anchor 1 (dog) must have 0.0 confidence after COCO remap
        self.assertLess(float(dets[0, 2, 5]), 1e-4)

    def test_verify_local_passes_on_starter_kit(self) -> None:
        self.assertTrue(verify_local.verify_repo(str(STARTER_DIR)))

    def test_verify_local_catches_bad_values_and_pt_files(self) -> None:
        # Pixel coordinates should fail
        bad_px = np.array([[[160.0, 120.0, 60.0, 80.0, 0.0, 0.9, 1.0]]], dtype=np.float32)
        fails, _ = verify_local._validate_output_values(bad_px)
        self.assertTrue(any("unnormalized" in f.lower() for f in fails))

        # Degrees instead of radians should fail
        bad_deg = np.array([[[0.5, 0.5, 0.2, 0.2, 45.0, 0.9, 1.0]]], dtype=np.float32)
        fails, _ = verify_local._validate_output_values(bad_deg)
        self.assertTrue(any("radians" in f.lower() for f in fails))

        # Class 2 (not 0 or 1) should fail
        bad_cls = np.array([[[0.5, 0.5, 0.2, 0.2, 0.1, 0.9, 2.0]]], dtype=np.float32)
        fails, _ = verify_local._validate_output_values(bad_cls)
        self.assertTrue(any("class ids" in f.lower() for f in fails))

        # Forbidden .pt at repo root should fail verify_repo
        repo = self.tmp / "bad_repo"
        (repo / "models").mkdir(parents=True)
        shutil.copy(STARTER_DIR / "model.json", repo / "model.json")
        shutil.copy(STARTER_DIR / "models" / "detector_int8.onnx", repo / "models" / "detector_int8.onnx")
        (repo / "models" / "best.pt").write_bytes(b"fake")
        self.assertFalse(verify_local.verify_repo(str(repo)))

    def test_train_coco_remap(self) -> None:
        lines = [
            "0 0.5 0.5 0.1 0.2",
            "2 0.3 0.4 0.2 0.1",
            "7 0.7 0.6 0.3 0.2",
            "16 0.2 0.2 0.1 0.1",  # dog -> dropped
        ]
        remapped = train.remap_yolo_label_lines(lines)
        self.assertEqual(
            remapped,
            [
                "0 0.5 0.5 0.1 0.2",
                "1 0.3 0.4 0.2 0.1",
                "1 0.7 0.6 0.3 0.2",
            ],
        )

    def test_score_local_runs_on_dev_clip_with_stub_and_perfect_scorer(self) -> None:
        # Bundled schema stub must score 0.0% mAP / 0.0% HOTA and 10/10 canary quiet
        res = score_local.score_repo(str(STARTER_DIR))
        self.assertEqual(res["map_50_95"], 0.0)
        self.assertEqual(res["hota"], 0.0)
        self.assertEqual(res["canary_quiet"], 10)
        # Verify dev_clip exists with real non-empty labels for both person and vehicle
        frames, gts, meta = score_local._load_clip(STARTER_DIR / "dev_clip")
        self.assertFalse(meta.get("in_official_test_suite", True))
        self.assertEqual(len(frames), 25)
        classes_seen = {int(g["cls"]) for f in gts for g in f}
        self.assertEqual(classes_seen, {0, 1})

    def test_verify_local_preview_deduplicates_boxes_via_nms(self) -> None:
        # 4 nearly identical overlapping boxes on one person must collapse to 1 after NMS
        raw = np.zeros((1, 6, 6), dtype=np.float32)
        for idx, (cx, conf) in enumerate([(160.0, 0.90), (161.0, 0.85), (159.0, 0.80), (160.5, 0.75)]):
            raw[0, 0:4, idx] = [cx, 160.0, 64.0, 96.0]
            raw[0, 4, idx] = conf
        raw_p = self.tmp / "raw_dup.onnx"
        wrap_p = self.tmp / "wrap_dup.onnx"
        _build_dummy_raw_yolo_onnx(raw_p, raw, imgsz=320, opset=17)
        export_onnx.wrap_yolo_onnx(raw_p, wrap_p, imgsz=320, num_classes=2, top_k=6)

        img_p = STARTER_DIR / "dev_clip" / "frames" / "frame_0000.jpg"
        sess = ort.InferenceSession(str(wrap_p), providers=["CPUExecutionProvider"])
        kept = verify_local._decode_and_draw_preview(
            str(img_p), sess, sess.get_inputs()[0].name, (320, 320), "letterbox", conf_thr=0.35, nms_iou=0.45
        )
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0]["conf"], 0.90, places=3)

    def test_wrap_yolox_coco80_multiplies_obj_and_cls_scores(self) -> None:
        # YOLOX layout: [1, N, 85] = [cx, cy, w, h, obj_conf, cls_0..cls_79]
        raw = np.zeros((1, 10, 85), dtype=np.float32)
        # Anchor 2: dog (COCO cls 16 -> ignored!) with obj=0.99, cls=0.99
        raw[0, 2, 0:4] = [100.0, 100.0, 50.0, 50.0]
        raw[0, 2, 4] = 0.99
        raw[0, 2, 5 + 16] = 0.99
        # Anchor 5: car (COCO cls 2 -> vehicle=1) with obj=0.80, cls=0.90 => conf=0.72
        raw[0, 5, 0:4] = [208.0, 104.0, 83.2, 41.6]
        raw[0, 5, 4] = 0.80
        raw[0, 5, 5 + 2] = 0.90
        # Anchor 8: person (COCO cls 0 -> person=0) with obj=0.50, cls=0.80 => conf=0.40
        raw[0, 8, 0:4] = [104.0, 208.0, 41.6, 83.2]
        raw[0, 8, 4] = 0.50
        raw[0, 8, 5 + 0] = 0.80

        raw_p = self.tmp / "raw_yolox.onnx"
        out_p = self.tmp / "wrapped_yolox.onnx"
        _build_dummy_raw_yolo_onnx(raw_p, raw, imgsz=416, opset=17)

        export_onnx.wrap_yolox_onnx(raw_p, out_p, imgsz=416, coco_remap=True, top_k=4)
        sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        dets = sess.run(None, {"images": np.zeros((1, 3, 416, 416), dtype=np.float32)})[0]
        self.assertEqual(dets.shape, (1, 4, 7))
        np.testing.assert_allclose(dets[0, 0, :4], [0.5, 0.25, 0.2, 0.1], atol=1e-3)
        self.assertAlmostEqual(float(dets[0, 0, 5]), 0.72, places=4)
        self.assertEqual(int(round(float(dets[0, 0, 6]))), 1)
        self.assertAlmostEqual(float(dets[0, 1, 5]), 0.40, places=4)
        self.assertEqual(int(round(float(dets[0, 1, 6]))), 0)
        self.assertLess(float(dets[0, 2, 5]), 1e-4)

    def test_train_prepare_coco_json(self) -> None:
        src_json = self.tmp / "instances_sample.json"
        dst_json = self.tmp / "wog_2class.json"
        src_json.write_text(
            json.dumps(
                {
                    "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
                    "annotations": [
                        {"id": 10, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 20]},  # person -> 1
                        {"id": 11, "image_id": 1, "category_id": 3, "bbox": [5, 6, 30, 15]},  # car -> 2
                        {"id": 12, "image_id": 2, "category_id": 18, "bbox": [2, 2, 8, 8]},  # dog -> dropped
                    ],
                }
            ),
            encoding="utf-8",
        )
        counts = train.prepare_coco_json(src_json, dst_json)
        self.assertEqual(counts, {"images": 1, "person": 1, "vehicle": 1})
        out = json.loads(dst_json.read_text(encoding="utf-8"))
        self.assertEqual([c["name"] for c in out["categories"]], ["person", "vehicle"])
        self.assertEqual([a["category_id"] for a in out["annotations"]], [1, 2])

    def test_verify_local_rejects_copyleft_license_and_accepts_apache(self) -> None:
        repo = self.tmp / "lic_repo"
        (repo / "models").mkdir(parents=True)
        shutil.copy(STARTER_DIR / "model.json", repo / "model.json")
        shutil.copy(STARTER_DIR / "models" / "detector_int8.onnx", repo / "models" / "detector_int8.onnx")

        (repo / "LICENSE").write_text("GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3\n", encoding="utf-8")
        self.assertFalse(verify_local.verify_repo(str(repo)))

        (repo / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")
        self.assertTrue(verify_local.verify_repo(str(repo)))


if __name__ == "__main__":
    unittest.main()
