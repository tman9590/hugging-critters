#!/usr/bin/env python3
"""Extract the selected classes and make a deterministic stratified split."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from hugging_critters.taxonomy import EASTERN_US_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Downloaded animals.zip")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--max-size", type=int, default=384,
                        help="Downsample the long edge once to avoid decoding huge web images every epoch")
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "validation"):
        for label in EASTERN_US_CLASSES:
            (args.output / split / label).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    counts: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(args.archive) as zf:
        for label in EASTERN_US_CLASSES:
            prefix = f"animals/{label}/"
            members = sorted(
                n for n in zf.namelist()
                if n.startswith(prefix) and not n.endswith("/")
            )
            if not members:
                raise RuntimeError(f"No images found for {label}")
            rng.shuffle(members)
            validation_count = max(1, round(len(members) * args.validation_fraction))
            validation = set(members[:validation_count])
            split_counts = {"train": 0, "validation": 0}
            for member in members:
                split = "validation" if member in validation else "train"
                target = args.output / split / label / Path(member).name
                with zf.open(member) as source:
                    image = ImageOps.exif_transpose(Image.open(BytesIO(source.read()))).convert("RGB")
                    image.thumbnail((args.max_size, args.max_size), Image.Resampling.LANCZOS)
                    image.save(target, "JPEG", quality=90, optimize=True)
                split_counts[split] += 1
            counts[label] = split_counts

    (args.output / "split.json").write_text(
        json.dumps({"seed": args.seed, "counts": counts}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {sum(sum(v.values()) for v in counts.values())} images")


if __name__ == "__main__":
    main()
