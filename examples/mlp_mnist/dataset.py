#!/usr/bin/env python3
"""Download MNIST and write train/test splits as GDDS shards."""

from __future__ import annotations

from pathlib import Path
import argparse
import gzip
import json
import struct
import sys
from typing import Iterator, Mapping
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gdds_utils import FieldSpec, TensorData, schema_hash, tensor, write_gdds_split  # noqa: E402


SERVER = "https://raw.githubusercontent.com/fgnt/mnist/master"
IMAGE_ROWS = 28
IMAGE_COLS = 28
IMAGE_PIXELS = IMAGE_ROWS * IMAGE_COLS
TRAIN_SAMPLES = 60_000
TEST_SAMPLES = 10_000

FIELDS = [
    FieldSpec("image", "f16", (IMAGE_PIXELS,)),
    FieldSpec("target", "i32", ()),
]

SPLITS = {
    "train": {
        "images": "train-images-idx3-ubyte.gz",
        "labels": "train-labels-idx1-ubyte.gz",
        "samples": TRAIN_SAMPLES,
    },
    "test": {
        "images": "t10k-images-idx3-ubyte.gz",
        "labels": "t10k-labels-idx1-ubyte.gz",
        "samples": TEST_SAMPLES,
    },
}


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using cached {path}")
        return path
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"Downloading {url}")
    urlretrieve(url, tmp)
    tmp.replace(path)
    print(f"Wrote {path}")
    return path


def read_exact(handle: gzip.GzipFile, nbytes: int, what: str) -> bytes:
    data = handle.read(nbytes)
    if len(data) != nbytes:
        raise ValueError(f"invalid MNIST file: truncated {what}; got {len(data)}, expected {nbytes}")
    return data


def load_images(path: Path, expected_samples: int) -> bytes:
    with gzip.open(path, "rb") as gz:
        magic, count, rows, cols = struct.unpack(">IIII", read_exact(gz, 16, "image header"))
        if magic != 2051:
            raise ValueError(f"invalid image file {path}: unexpected magic {magic}")
        if count != expected_samples:
            raise ValueError(f"invalid image file {path}: expected {expected_samples} images, got {count}")
        if rows != IMAGE_ROWS or cols != IMAGE_COLS:
            raise ValueError(f"invalid image file {path}: expected 28x28 images, got {rows}x{cols}")
        payload = read_exact(gz, expected_samples * IMAGE_PIXELS, "image payload")
    return payload


def load_labels(path: Path, expected_samples: int) -> bytes:
    with gzip.open(path, "rb") as gz:
        magic, count = struct.unpack(">II", read_exact(gz, 8, "label header"))
        if magic != 2049:
            raise ValueError(f"invalid label file {path}: unexpected magic {magic}")
        if count != expected_samples:
            raise ValueError(f"invalid label file {path}: expected {expected_samples} labels, got {count}")
        payload = read_exact(gz, expected_samples, "label payload")
    for value in payload:
        if value > 9:
            raise ValueError(f"invalid label file {path}: label out of range: {value}")
    return payload


def pixel_lookup_tables() -> tuple[bytes, bytes]:
    low = bytearray(256)
    high = bytearray(256)
    for value in range(256):
        packed = struct.pack("<e", float(value) / 255.0)
        low[value] = packed[0]
        high[value] = packed[1]
    return bytes(low), bytes(high)


def image_to_f16(image: bytes, low_table: bytes, high_table: bytes) -> bytes:
    if len(image) != IMAGE_PIXELS:
        raise ValueError(f"expected {IMAGE_PIXELS} image bytes, got {len(image)}")
    out = bytearray(IMAGE_PIXELS * 2)
    out[0::2] = image.translate(low_table)
    out[1::2] = image.translate(high_table)
    return bytes(out)


def mnist_samples(images: bytes, labels: bytes, limit: int) -> Iterator[Mapping[str, TensorData]]:
    count = len(labels)
    if len(images) != count * IMAGE_PIXELS:
        raise ValueError("image/label count mismatch")
    if limit <= 0 or limit > count:
        limit = count
    low_table, high_table = pixel_lookup_tables()
    label_tensors = [tensor(label, "i32") for label in range(10)]
    for index in range(limit):
        offset = index * IMAGE_PIXELS
        image = images[offset : offset + IMAGE_PIXELS]
        label = labels[index]
        yield {
            "image": TensorData("f16", (IMAGE_PIXELS,), image_to_f16(image, low_table, high_table)),
            "target": label_tensors[label],
        }


def remove_split_shards(out_dir: Path, split: str) -> None:
    for path in out_dir.glob(f"{split}*.gdds"):
        if path.name == f"{split}.gdds" or path.name.startswith(f"{split}-"):
            path.unlink()


def write_manifest(out_dir: Path, split_counts: Mapping[str, int], split_paths: Mapping[str, list[Path]]) -> None:
    manifest = {
        "format": "GDDS",
        "version": 1,
        "schema_hash": f"0x{schema_hash(FIELDS):016x}",
        "fields": [
            {"name": field.name, "dtype": field.dtype, "shape": list(field.shape)}
            for field in FIELDS
        ],
        "normalization": {
            "image": "flattened 28x28 pixels as f16, value = uint8 / 255.0",
            "target": "int32 class id in [0, 9]",
        },
        "splits": {
            split: {
                "samples": split_counts[split],
                "shards": [path.name for path in split_paths[split]],
            }
            for split in split_paths
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def positive_limit(value: int, max_value: int, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if value == 0:
        return max_value
    if value > max_value:
        raise ValueError(f"{name}={value} exceeds available samples ({max_value})")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--samples-per-shard", type=int, default=8192)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means all 60000 training samples")
    parser.add_argument("--test-limit", type=int, default=0, help="0 means all 10000 test samples")
    args = parser.parse_args()

    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    out_dir = args.out_dir
    cache_dir = args.cache_dir if args.cache_dir is not None else out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    limits = {
        "train": positive_limit(args.train_limit, TRAIN_SAMPLES, "--train-limit"),
        "test": positive_limit(args.test_limit, TEST_SAMPLES, "--test-limit"),
    }
    split_counts: dict[str, int] = {}
    split_paths: dict[str, list[Path]] = {}

    for split, spec in SPLITS.items():
        expected = int(spec["samples"])
        image_path = download(f"{SERVER}/{spec['images']}", cache_dir / str(spec["images"]))
        label_path = download(f"{SERVER}/{spec['labels']}", cache_dir / str(spec["labels"]))
        images = load_images(image_path, expected)
        labels = load_labels(label_path, expected)
        remove_split_shards(out_dir, split)
        print(f"Writing {split} GDDS split ({limits[split]} samples)")
        paths = write_gdds_split(
            out_dir,
            split,
            FIELDS,
            mnist_samples(images, labels, limits[split]),
            samples_per_shard=args.samples_per_shard,
            write_manifest=False,
        )
        split_counts[split] = limits[split]
        split_paths[split] = paths
        for path in paths:
            print(path)

    write_manifest(out_dir, split_counts, split_paths)
    print(out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
