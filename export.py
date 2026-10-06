#!/usr/bin/env python3
"""Export one trained classifier to every Scrypted custom-model backend."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import torch

from train import MEAN, STD, make_model


def write_config(path: Path, files: list[str], labels: dict[str, str]) -> None:
    config = {
        "input_shape": [1, 3, 224, 224],
        # Scrypted currently uses this discriminator for image classifiers.
        "model": "resnet",
        "mean": MEAN,
        "std": STD,
        "files": files,
        "labels": labels,
    }
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--pnnx", default="pnnx", help="pnnx executable for NCNN export")
    args = parser.parse_args()
    labels = json.loads((args.checkpoint_dir / "classmap.json").read_text())
    model = make_model(len(labels), pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint_dir / "best_model.pth", map_location="cpu", weights_only=True))
    model.eval()
    example = torch.zeros(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)

    if args.output.exists():
        shutil.rmtree(args.output)
    onnx_dir = args.output / "onnx"
    onnx_dir.mkdir(parents=True)
    torch.onnx.export(model, example, onnx_dir / "model.onnx", input_names=["input"],
                      output_names=["out0"], opset_version=17, dynamo=False)
    write_config(onnx_dir, ["model.onnx"], labels)

    import openvino as ov
    openvino_dir = args.output / "openvino"
    openvino_dir.mkdir(parents=True)
    ov_model = ov.convert_model(str(onnx_dir / "model.onnx"))
    ov.save_model(ov_model, openvino_dir / "model.xml", compress_to_fp16=True)
    write_config(openvino_dir, ["model.xml", "model.bin"], labels)

    import coremltools as ct
    coreml_dir = args.output / "coreml"
    coreml_dir.mkdir(parents=True)
    # The neuralnetwork protobuf can be written on Windows/Linux. Packaging an
    # mlprogram requires Apple's macOS-only BlobWriter even though conversion
    # itself is cross-platform.
    coreml_model = ct.convert(traced, convert_to="neuralnetwork",
                              inputs=[ct.TensorType(name="input", shape=example.shape)])
    package = coreml_dir / "model.mlpackage"
    model_dir = package / "Data" / "com.apple.CoreML"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "model.mlmodel"
    ct.models.utils.save_spec(coreml_model.get_spec(), str(model_file))
    model_id = str(uuid.uuid4())
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            model_id: {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel",
            }
        },
        "rootModelIdentifier": model_id,
    }
    (package / "Manifest.json").write_text(json.dumps(manifest, indent=4) + "\n")
    write_config(coreml_dir, [
        "model.mlpackage/Manifest.json",
        "model.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    ], labels)

    ncnn_dir = args.output / "ncnn"
    ncnn_dir.mkdir(parents=True)
    traced.save(ncnn_dir / "model.pt")
    subprocess.run([args.pnnx, str(ncnn_dir / "model.pt"), "inputshape=[1,3,224,224]"], check=True)
    (ncnn_dir / "model.pt").unlink()
    for generated in ("model.pnnx.bin", "model.pnnx.param", "model.pnnx.onnx",
                      "model_pnnx.py", "model_ncnn.py"):
        candidate = ncnn_dir / generated
        if candidate.exists():
            candidate.unlink()
    write_config(ncnn_dir, ["model.ncnn.param", "model.ncnn.bin"], labels)


if __name__ == "__main__":
    main()
