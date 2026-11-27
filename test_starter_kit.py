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


def _make_nonzero_wrapped_detector(out_path: Path, raw_p: Path, imgsz: int = 320) -> None:
    raw = np.zeros((1, 6, 4), dtype=np.float32)
    raw[0, 0:4, 0] = [160.0, 160.0, 64.0, 96.0]
    raw[0, 4, 0] = 0.85
    raw[0, 5, 0] = 0.05
    _build_dummy_raw_yolo_onnx(raw_p, raw, imgsz=imgsz, opset=17)
    export_onnx.wrap_yolo_onnx(raw_p, out_path, imgsz=imgsz, num_classes=2, top_k=4)


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

    def test_verify_local_rejects_bundled_stub_and_passes_real_detector(self) -> None:
        # Bundled schema stub must be rejected so new users do not waste a Run Token on a 0% model
        self.assertFalse(verify_local.verify_repo(str(STARTER_DIR)))

        repo = self.tmp / "real_repo"
        (repo / "models").mkdir(parents=True)
        shutil.copy(STARTER_DIR / "model.json", repo / "model.json")
        shutil.copy(STARTER_DIR / "LICENSE", repo / "LICENSE")
        shutil.copytree(STARTER_DIR / "dev_clip", repo / "dev_clip")
        _make_nonzero_wrapped_detector(repo / "models" / "detector_int8.onnx", self.tmp / "raw_ok.onnx")
        self.assertTrue(verify_local.verify_repo(str(repo)))

    def test_verify_local_rejects_zero_detection_model_on_real_frame(self) -> None:
        # Non-stub graph name, yet confidence is always < conf_threshold -> 0 detections on dev_clip
        raw = np.zeros((1, 6, 4), dtype=np.float32)
        raw[0, 0:4, 0] = [160.0, 160.0, 64.0, 96.0]
        raw[0, 4, 0] = 0.05  # below conf_threshold=0.35
        raw_p = self.tmp / "raw_low.onnx"
        _build_dummy_raw_yolo_onnx(raw_p, raw, imgsz=320, opset=17)

        repo = self.tmp / "low_repo"
        (repo / "models").mkdir(parents=True)
        shutil.copy(STARTER_DIR / "model.json", repo / "model.json")
        shutil.copy(STARTER_DIR / "LICENSE", repo / "LICENSE")
        shutil.copytree(STARTER_DIR / "dev_clip", repo / "dev_clip")
        export_onnx.wrap_yolo_onnx(raw_p, repo / "models" / "detector_int8.onnx", imgsz=320, num_classes=2, top_k=4)
        self.assertFalse(verify_local.verify_repo(str(repo)))

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
        _make_nonzero_wrapped_detector(repo / "models" / "detector_int8.onnx", self.tmp / "raw_lic.onnx")

        (repo / "LICENSE").write_text("GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3\n", encoding="utf-8")
        self.assertFalse(verify_local.verify_repo(str(repo)))

        (repo / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004\n", encoding="utf-8")
        self.assertTrue(verify_local.verify_repo(str(repo)))

    def test_wrap_yolox_bakes_bgr255_preprocess_and_grid_decode(self) -> None:
        # 416x416 undecoded YOLOX has N = 52*52 + 26*26 + 13*13 = 3549 anchors
        # Anchor 0 is stride=8, grid=(0, 0). Anchor index 2704 (= 52*52) is stride=16, grid=(0, 0).
        inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 416, 416])
        out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3549, 85])

        # Compute mean of inner input[0, :, :, :] -> [1, 3] representing BGR [0..255] if wrapped with bgr255
        # Use channel 0 (B -> must become 204.0 when R=0.2, G=0.5, B=0.8) / 204.0 to gate objectness!
        base = np.zeros((1, 3549, 85), dtype=np.float32)
        # Put undecoded offsets on anchor 0 (stride=8, grid=(0,0)), strictly inside [0, 416] so no edge clamp triggers:
        # tx=10.0, ty=10.0 -> cx_px = (10+0)*8 = 80.0, cy_px = (10+0)*8 = 80.0
        # tw=ln(4), th=ln(8) -> w_px = 4*8 = 32.0, h_px = 8*8 = 64.0
        base[0, 0, 0:4] = [10.0, 10.0, math.log(4.0), math.log(8.0)]
        base[0, 0, 4] = 1.0  # will be scaled by (ch0 / 204.0) * 0.90
        base[0, 0, 5 + 2] = 0.90  # car -> vehicle=1

        ch0_starts = numpy_helper.from_array(np.array([0], dtype=np.int64), name="s0")
        ch0_ends = numpy_helper.from_array(np.array([1], dtype=np.int64), name="e1")
        ch0_axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="a1")
        ch0_div = numpy_helper.from_array(np.array([255.0], dtype=np.float32), name="d255")
        base_t = numpy_helper.from_array(base, name="base_arr")
        sh_t = numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), name="sh111")

        # Anchor 0 confidence directly mirrors inner channel 0 mean / 255.0:
        # for RGB (0.2, 0.5, 0.8), BGR*255 gives inner ch0 = 0.8*255 = 204 -> gate = 0.80 -> conf = 0.80*0.90 = 0.72!
        s_box_s = numpy_helper.from_array(np.array([0], dtype=np.int64), name="bs0")
        s_box_e = numpy_helper.from_array(np.array([4], dtype=np.int64), name="be4")
        s_rest_e = numpy_helper.from_array(np.array([85], dtype=np.int64), name="re85")
        ax2 = numpy_helper.from_array(np.array([2], dtype=np.int64), name="ax2")
        nodes = [
            helper.make_node("Slice", ["images", "s0", "e1", "a1"], ["ch0"]),
            helper.make_node("ReduceMean", ["ch0"], ["m0"], keepdims=0),
            helper.make_node("Reshape", ["m0", "sh111"], ["m3d"]),
            helper.make_node("Div", ["m3d", "d255"], ["gate"]),
            helper.make_node("Slice", ["base_arr", "bs0", "be4", "ax2"], ["b_coords"]),
            helper.make_node("Slice", ["base_arr", "be4", "re85", "ax2"], ["b_scores"]),
            helper.make_node("Mul", ["b_scores", "gate"], ["g_scores"]),
            helper.make_node("Concat", ["b_coords", "g_scores"], ["output0"], axis=2),
        ]
        graph = helper.make_graph(
            nodes,
            "yolox_raw_undecoded",
            [inp],
            [out],
            initializer=[ch0_starts, ch0_ends, ch0_axes, ch0_div, base_t, sh_t, s_box_s, s_box_e, s_rest_e, ax2],
        )
        raw_model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 17)])
        raw_p = self.tmp / "raw_yolox_3549.onnx"
        out_p = self.tmp / "wrapped_yolox_3549.onnx"
        onnx.save(raw_model, str(raw_p))

        export_onnx.wrap_yolox_onnx(raw_p, out_p, imgsz=416, coco_remap=True, top_k=4)
        sess = ort.InferenceSession(str(out_p), providers=["CPUExecutionProvider"])
        rgb = np.empty((1, 3, 416, 416), dtype=np.float32)
        rgb[:, 0] = 0.2  # R
        rgb[:, 1] = 0.5  # G
        rgb[:, 2] = 0.8  # B -> inner ch0 must become 0.8 * 255 = 204
        dets = sess.run(None, {"images": rgb})[0]
        self.assertEqual(dets.shape, (1, 4, 7))
        # Box normalized to [416, 416]: cx=80/416, cy=80/416, w=32/416, h=64/416
        np.testing.assert_allclose(
            dets[0, 0, :4],
            [80.0 / 416.0, 80.0 / 416.0, 32.0 / 416.0, 64.0 / 416.0],
            atol=1e-4,
        )
        # Dummy graph multiplies both obj_conf (1.0) and cls_2 (0.90) by gate=(B*255)/255=0.8,
        # and YOLOX computes conf = obj_conf * cls_prob = (0.8) * (0.90 * 0.8) = 0.576.
        self.assertAlmostEqual(float(dets[0, 0, 5]), 0.8 * 0.8 * 0.90, places=4)
        self.assertEqual(int(round(float(dets[0, 0, 6]))), 1)

        # Flip B to 0.2 -> inner ch0 drops from 204 to 51, dropping conf to 0.2 * 0.2 * 0.90 = 0.036
        rgb_low_b = rgb.copy()
        rgb_low_b[:, 0] = 0.8
        rgb_low_b[:, 2] = 0.2
        dets_low_b = sess.run(None, {"images": rgb_low_b})[0]
        self.assertAlmostEqual(float(dets_low_b[0, 0, 5]), 0.2 * 0.2 * 0.90, places=4)


if __name__ == "__main__":
    unittest.main()
