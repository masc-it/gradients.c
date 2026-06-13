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

from gdds_utils import (  # noqa: E402
    GDDS_DEFAULT_MAX_SHARD_BYTES,
    GDDS_RECORD_FIELD_DESC_SIZE,
    GDDS_RECORD_HEADER_SIZE,
    FieldSpec,
    GddsSplitWriter,
    TensorData,
    field_metadata,
    max_fixed_records_per_shard,
    schema_hash,
    tensor,
)


SERVER = "https://raw.githubusercontent.com/fgnt/mnist/master"
IMAGE_ROWS = 28
IMAGE_COLS = 28
IMAGE_PIXELS = IMAGE_ROWS * IMAGE_COLS
TRAIN_SAMPLES = 60_000
TEST_SAMPLES = 10_000
DATA_FORMAT_VERSION = "mnist-u8-v1"

FIELDS = [
    FieldSpec("image", "u8", (IMAGE_PIXELS,), collate="stack"),
    FieldSpec("target", "i32", (), collate="stack"),
]
MNIST_RECORD_NBYTES = GDDS_RECORD_HEADER_SIZE + 2 * GDDS_RECORD_FIELD_DESC_SIZE + IMAGE_PIXELS + 4

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


def mnist_samples(images: bytes, labels: bytes, limit: int) -> Iterator[Mapping[str, TensorData]]:
    count = len(labels)
    if len(images) != count * IMAGE_PIXELS:
        raise ValueError("image/label count mismatch")
    if limit <= 0 or limit > count:
        limit = count
    label_tensors = [tensor(label, "i32") for label in range(10)]
    for index in range(limit):
        offset = index * IMAGE_PIXELS
        image = images[offset : offset + IMAGE_PIXELS]
        label = labels[index]
        yield {
            "image": TensorData("u8", (IMAGE_PIXELS,), image),
            "target": label_tensors[label],
        }


def split_shards(out_dir: Path, split: str) -> list[Path]:
    return sorted(
        path
        for path in out_dir.glob(f"{split}*.gdds")
        if path.name == f"{split}.gdds" or path.name.startswith(f"{split}-")
    )


def remove_split_shards(out_dir: Path, split: str) -> None:
    for path in split_shards(out_dir, split):
        path.unlink()


def expected_shard_samples(n_samples: int, max_shard_bytes: int) -> list[int]:
    shard_limit = max_fixed_records_per_shard(FIELDS, MNIST_RECORD_NBYTES, max_shard_bytes)
    if shard_limit <= 0:
        raise ValueError("MNIST records do not fit in max_shard_bytes")
    counts: list[int] = []
    remaining = n_samples
    while remaining > 0:
        count = min(shard_limit, remaining)
        counts.append(count)
        remaining -= count
    return counts


def read_gdds_header(path: Path) -> tuple[int, str]:
    with path.open("rb") as handle:
        header = handle.read(128)
    if len(header) != 128:
        raise ValueError(f"{path} is not a complete GDDS shard")
    magic = header[:8]
    version, header_size = struct.unpack_from("<II", header, 8)
    if magic != b"GDDSv1\0\0" or version != 1 or header_size != 128:
        raise ValueError(f"{path} is not a GDDS v1 shard")
    n_samples = struct.unpack_from("<Q", header, 24)[0]
    shard_schema_hash = struct.unpack_from("<Q", header, 64)[0]
    return n_samples, f"0x{shard_schema_hash:016x}"


def existing_gdds_ready(
    out_dir: Path,
    limits: Mapping[str, int],
    max_shard_bytes: int,
) -> bool:
    expected_schema_hash = f"0x{schema_hash(FIELDS):016x}"
    split_paths: dict[str, list[Path]] = {}
    try:
        for split, n_samples in limits.items():
            expected_counts = expected_shard_samples(n_samples, max_shard_bytes)
            expected_names = [f"{split}-{index:05d}.gdds" for index in range(len(expected_counts))]
            actual_paths = split_shards(out_dir, split)
            if [path.name for path in actual_paths] != expected_names:
                return False
            for path, expected_count in zip(actual_paths, expected_counts):
                shard_samples, shard_schema_hash = read_gdds_header(path)
                if shard_samples != expected_count or shard_schema_hash != expected_schema_hash:
                    return False
            split_paths[split] = actual_paths
    except (OSError, ValueError, struct.error):
        return False

    write_manifest(out_dir, limits, split_paths, max_shard_bytes)
    print(f"Using existing GDDS dataset in {out_dir}")
    return True


def write_manifest(
    out_dir: Path,
    split_counts: Mapping[str, int],
    split_paths: Mapping[str, list[Path]],
    max_shard_bytes: int,
) -> None:
    manifest = {
        "format": "GDDS",
        "version": 1,
        "schema_hash": f"0x{schema_hash(FIELDS):016x}",
        "fields": [field_metadata(field) for field in FIELDS],
        "storage": {
            "image": "flattened 28x28 pixels as raw uint8 in [0, 255]",
            "target": "int32 class id in [0, 9]",
        },
        "runtime_transform": {
            "image": "main.c normalizes uint8 to f16 with value = uint8 / 255.0",
        },
        "splits": {
            split: {
                "samples": split_counts[split],
                "shards": [path.name for path in split_paths[split]],
            }
            for split in split_paths
        },
        "prep": {
            "format_version": DATA_FORMAT_VERSION,
            "max_shard_bytes": max_shard_bytes,
            "train_limit": split_counts.get("train", 0),
            "test_limit": split_counts.get("test", 0),
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


def write_mnist_split(
    out_dir: Path,
    split: str,
    images: bytes,
    labels: bytes,
    limit: int,
    max_shard_bytes: int,
) -> list[Path]:
    writer = GddsSplitWriter(
        out_dir,
        split,
        FIELDS,
        max_shard_bytes=max_shard_bytes,
    )
    try:
        for sample in mnist_samples(images, labels, limit):
            writer.write_sample(sample)
        return writer.finish()
    except BaseException:
        writer.abort()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-shard-bytes", type=int, default=GDDS_DEFAULT_MAX_SHARD_BYTES)
    parser.add_argument("--train-limit", type=int, default=0, help="0 means all 60000 training samples")
    parser.add_argument("--test-limit", type=int, default=0, help="0 means all 10000 test samples")
    args = parser.parse_args()

    if args.max_shard_bytes <= 0:
        raise ValueError("--max-shard-bytes must be positive")
    out_dir = args.out_dir
    cache_dir = args.cache_dir if args.cache_dir is not None else out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    limits = {
        "train": positive_limit(args.train_limit, TRAIN_SAMPLES, "--train-limit"),
        "test": positive_limit(args.test_limit, TEST_SAMPLES, "--test-limit"),
    }
    if existing_gdds_ready(out_dir, limits, args.max_shard_bytes):
        return 0

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
        paths = write_mnist_split(
            out_dir,
            split,
            images,
            labels,
            limits[split],
            args.max_shard_bytes,
        )
        split_counts[split] = limits[split]
        split_paths[split] = paths
        for path in paths:
            print(path)

    write_manifest(out_dir, split_counts, split_paths, args.max_shard_bytes)
    print(out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
