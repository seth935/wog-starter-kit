"""Generates an untrained schema smoke-test ONNX stub for the Weights of Giants starter kit.

Input:  images [1, 3, 320, 320] float32
Output: detections [1, 100, 7] float32: [cx, cy, w, h, theta, conf, cls] (all zeros)
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

    input_tensor = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 320, 320])
    output_tensor = helper.make_tensor_value_info("detections", TensorProto.FLOAT, [1, 100, 7])

    # Schema smoke-test stub: outputs zeros ([1, 100, 7]) so it validates tensor
    # shapes and ONNX Runtime wiring without emitting hardcoded boxes.
    baseline_dets = np.zeros((1, 100, 7), dtype=np.float32)

    conv_w = np.zeros((1, 3, 1, 1), dtype=np.float32)
    conv_b = np.zeros((1,), dtype=np.float32)

    conv_node = helper.make_node("Conv", inputs=["images", "conv_w", "conv_b"], outputs=["conv_out"])
    pool_node = helper.make_node("GlobalAveragePool", inputs=["conv_out"], outputs=["pool_out"])
    reshape_node = helper.make_node("Reshape", inputs=["pool_out", "target_shape"], outputs=["gate_3d"])
    out_node = helper.make_node("Add", inputs=["zero_dets", "gate_3d"], outputs=["detections"])

    graph = helper.make_graph(
        [conv_node, pool_node, reshape_node, out_node],
        "wog_cortex_a53_schema_stub",
        [input_tensor],
        [output_tensor],
        initializer=[
            helper.make_tensor("conv_w", TensorProto.FLOAT, [1, 3, 1, 1], conv_w.flatten().tolist()),
            helper.make_tensor("conv_b", TensorProto.FLOAT, [1], conv_b.flatten().tolist()),
            helper.make_tensor("target_shape", TensorProto.INT64, [3], [1, 1, 1]),
            helper.make_tensor("zero_dets", TensorProto.FLOAT, [1, 100, 7], baseline_dets.flatten().tolist()),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="weights_of_giants_arena",
        ir_version=8,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Created schema smoke-test ONNX stub at: {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    create_baseline_onnx(str(Path(__file__).parent / "detector_int8.onnx"))
