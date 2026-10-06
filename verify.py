#!/usr/bin/env python3
"""Check package structure and numerical agreement of available runtimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("models"))
    args = parser.parse_args()
    sample = np.random.default_rng(20261005).normal(size=(1, 3, 224, 224)).astype(np.float32)
    expected_labels = None
    for backend in ("onnx", "openvino", "coreml", "ncnn"):
        config_path = args.models / backend / "config.json"
        config = json.loads(config_path.read_text())
        assert config["input_shape"] == [1, 3, 224, 224]
        assert config["model"] == "resnet"
        assert sorted(map(int, config["labels"])) == list(range(len(config["labels"])))
        expected_labels = expected_labels or config["labels"]
        assert config["labels"] == expected_labels
        for relative in config["files"]:
            assert (config_path.parent / relative).is_file(), f"Missing {backend}/{relative}"

    import onnxruntime as ort
    onnx_session = ort.InferenceSession(str(args.models / "onnx" / "model.onnx"))
    onnx_output = onnx_session.run(None, {onnx_session.get_inputs()[0].name: sample})[0]
    import openvino as ov
    compiled = ov.Core().compile_model(str(args.models / "openvino" / "model.xml"), "CPU")
    openvino_output = compiled([sample])[compiled.output(0)]
    np.testing.assert_allclose(onnx_output, openvino_output, rtol=2e-2, atol=2e-2)

    import coremltools as ct
    coreml_spec = ct.models.utils.load_spec(str(
        args.models / "coreml" / "model.mlpackage" / "Data" / "com.apple.CoreML" / "model.mlmodel"
    ))
    assert list(coreml_spec.description.input[0].type.multiArrayType.shape) == [1, 3, 224, 224]
    ncnn_param = (args.models / "ncnn" / "model.ncnn.param").read_text()
    assert " out0 " in ncnn_param
    print(f"Verified 4 manifests, {len(expected_labels)} labels, all model structures, "
          "and ONNX/OpenVINO numerical agreement")


if __name__ == "__main__":
    main()
