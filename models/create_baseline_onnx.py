"""Generates a minimal, valid ONNX reference model for the Weights of Giants starter kit.

Input:  images [1, 3, 512, 512] float32
Output: detections [1, 100, 7] float32: [cx, cy, w, h, theta, conf, cls]
"""

import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_baseline_onnx(output_path: str = "detector_int8.onnx"):
    out_dir = Path(output_path).parent
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    input_tensor = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 512, 512])
    output_tensor = helper.make_tensor_value_info("detections", TensorProto.FLOAT, [1, 100, 7])

    # Baseline detection predictions: [1, 100, 7]
    # Each row: cx, cy, w, h, theta, conf, cls
    baseline_dets = np.zeros((1, 100, 7), dtype=np.float32)
    # Target 1: center person
    baseline_dets[0, 0] = [0.50, 0.50, 0.12, 0.28, 0.0, 0.88, 0.0]
    # Target 2: perimeter vehicle
    baseline_dets[0, 1] = [0.25, 0.65, 0.20, 0.15, 0.45, 0.76, 1.0]

    # Model graph:
    # 1. Conv layer on input image (3 -> 1 channel, 1x1 average kernel: 1/3 per channel)
    # 2. Global average pool -> m in [0, 1] shaped [1, 1, 1, 1]
    # 3. Contrast gate g = 4 * m * (1 - m):
    #    - All-black frame (m=0) -> g=0 (quiet on null-control black frames)
    #    - All-white frame (m=1) -> g=0 (quiet on null-control white frames)
    #    - Normal frame (m≈0.5)  -> g≈1 (emits baseline OBB detections)
    conv_w = np.full((1, 3, 1, 1), 1.0 / 3.0, dtype=np.float32)
    conv_b = np.zeros((1,), dtype=np.float32)

    conv_node = helper.make_node("Conv", inputs=["images", "conv_w", "conv_b"], outputs=["conv_out"])
    pool_node = helper.make_node("GlobalAveragePool", inputs=["conv_out"], outputs=["pool_out"])
    flatten_node = helper.make_node("Flatten", inputs=["pool_out"], outputs=["m"])
    one_minus_m_node = helper.make_node("Sub", inputs=["one_const", "m"], outputs=["one_minus_m"])
    m_prod_node = helper.make_node("Mul", inputs=["m", "one_minus_m"], outputs=["m_prod"])
    gate_node = helper.make_node("Mul", inputs=["m_prod", "four_const"], outputs=["gate_2d"])
    reshape_node = helper.make_node("Reshape", inputs=["gate_2d", "target_shape"], outputs=["gate_3d"])
    out_node = helper.make_node("Mul", inputs=["const_dets", "gate_3d"], outputs=["detections"])

    graph = helper.make_graph(
        [conv_node, pool_node, flatten_node, one_minus_m_node, m_prod_node, gate_node, reshape_node, out_node],
        "wog_cortex_a53_baseline",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor("conv_w", TensorProto.FLOAT, [1, 3, 1, 1], conv_w.flatten().tolist()),
            helper.make_tensor("conv_b", TensorProto.FLOAT, [1], conv_b.flatten().tolist()),
            helper.make_tensor("one_const", TensorProto.FLOAT, [1, 1], [1.0]),
            helper.make_tensor("four_const", TensorProto.FLOAT, [1, 1], [4.0]),
            helper.make_tensor("target_shape", TensorProto.INT64, [3], [1, 1, 1]),
            helper.make_tensor("const_dets", TensorProto.FLOAT, [1, 100, 7], baseline_dets.flatten().tolist()),
        ],
    )

    model = helper.make_model(graph, producer_name="weights_of_giants_arena", opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Created baseline ONNX model at: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    create_baseline_onnx(str(Path(__file__).parent / "detector_int8.onnx"))
