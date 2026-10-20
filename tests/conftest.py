from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from recognition_accuracy.alignment import REFERENCE


def save_model(path: Path, nodes: list, inputs: list, outputs: list, initializers=()) -> Path:
    graph = helper.make_graph(nodes, "test", inputs, outputs, initializer=list(initializers))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


@pytest.fixture
def detector_model(tmp_path):
    nodes, output_infos = [], []
    for stride in (32, 16, 8):
        h = 128 // stride
        maps = [np.zeros((1, channels, h, h), dtype=np.float32) for channels in (4, 8, 20)]
        maps[0][:, :2] = 1
        if stride == 8:
            maps[0][0, 2, 7, 7] = 0.99
            deltas = ((REFERENCE * (128 / 112) - 63.5) / 32).ravel()
            maps[2][0, :10, 7, 7] = deltas
        for label, value in zip(("cls", "box", "landmark"), maps):
            name = f"{label}_{stride}"
            nodes.append(
                helper.make_node(
                    "Constant",
                    [],
                    [name],
                    value=numpy_helper.from_array(value),
                )
            )
            output_infos.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, value.shape))
    return save_model(
        tmp_path / "detector.onnx",
        nodes,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 128, 128])],
        output_infos,
    )


@pytest.fixture
def recognition_model(tmp_path):
    return save_model(
        tmp_path / "mobilefacenet.onnx",
        [helper.make_node("ReduceMean", ["input"], ["embedding"], axes=[1, 2], keepdims=0)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 112, 112, 3])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 3])],
    )


@pytest.fixture
def nchw_recognition_model(tmp_path):
    return save_model(
        tmp_path / "recognition-nchw.onnx",
        [helper.make_node("ReduceMean", ["input"], ["embedding"], axes=[2, 3], keepdims=0)],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 112, 112])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 3])],
    )
