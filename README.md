# Hugging Critters

An Eastern U.S. animal image classifier packaged for every Scrypted backend that
currently supports custom classifier models: **ONNX, OpenVINO, CoreML, and NCNN**.

The model is fine-tuned from ImageNet MobileNetV3-Small on the public-domain
[Fr0styKn1ght/Animals](https://huggingface.co/datasets/Fr0styKn1ght/Animals)
dataset. The source has 5,700 images and 90 broad classes. This project selects
the 64 classes with a native, established, introduced, or domestic occurrence in
the eastern contiguous United States or its Atlantic/Gulf coastal waters. The
exact inclusion rule, included labels, and reasons for exclusions are versioned
in [`hugging_critters/taxonomy.py`](hugging_critters/taxonomy.py).

The committed run reached **83.3% top-1 accuracy** on the deterministic held-out
set of 768 images across 64 classes, with the best checkpoint at epoch 10.

## Install in Scrypted

1. Install the object-detection plugin appropriate for the machine: ONNX,
   OpenVINO, CoreML, or NCNN.
2. Open that plugin, choose **Add New**, and select its custom model option.
3. Name it `Hugging Critters` and enter this repository URL.
4. In Scrypted NVR, choose Hugging Critters as the animal classifier.

Scrypted resolves the repository URL to
`models/<backend>/config.json`, downloads the files listed there, and feeds the
animal crop to this classifier. TensorFlow Lite is not included because
Scrypted's TensorFlow Lite plugin does not expose the custom-model device API;
there is currently no compatible manifest target for a third-party classifier.

## Reproduce

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
curl -L -o animals.zip \
  https://huggingface.co/datasets/Fr0styKn1ght/Animals/resolve/main/animals.zip
python prepare_dataset.py animals.zip
python train.py
python export.py --pnnx /path/to/pnnx
python verify.py
```

The split is deterministic and stratified. Training uses label smoothing,
ImageNet normalization, geometric/color augmentation, AdamW, and cosine learning
rate decay. See `checkpoints/metrics.json` for the committed run's held-out
result. Dataset images and training checkpoints are intentionally not committed.

## Model layout

```text
models/
  coreml/config.json + model.mlpackage/
  ncnn/config.json + model.ncnn.param + model.ncnn.bin
  onnx/config.json + model.onnx
  openvino/config.json + model.xml + model.bin
```

## Limitations

This is a small, coarse-label source dataset and contains mostly web imagery,
not infrared security-camera crops. Treat the validation metric as an internal
split result, then tune Scrypted's threshold against real camera footage. Broad
labels (for example `bear`, `owl`, or `snake`) are not species-level IDs.

Code and model exports are MIT licensed. The source dataset is marked Unlicense
on Hugging Face; users should independently confirm upstream image provenance
for their intended use.
