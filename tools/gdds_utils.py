#!/usr/bin/env python3
"""Utilities for writing GDDS v1 dataset shards.

GDDS is the generic mmap-friendly dataset format used by gradients.c.  A GDDS
shard is self describing: a fixed binary header, a fixed-width field schema, a
random-access record index, and record payloads.  The C runtime maps shard files
and materializes only the samples requested by the dataloader workers.

This module intentionally has no third-party dependencies so dataset preparation
scripts can reuse it in small examples and production preprocessing jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
import json
import math
import os
import struct

GDDS_MAGIC = b"GDDSv1\0\0"
GDDS_RECORD_MAGIC = b"GDDR"
GDDS_VERSION = 1
GDDS_HEADER_SIZE = 128
GDDS_FIELD_NAME_MAX = 64
GDDS_FIELD_DESC_SIZE = 160
GDDS_INDEX_ENTRY_SIZE = 16
GDDS_RECORD_HEADER_SIZE = 20
GDDS_RECORD_FIELD_DESC_SIZE = 88
GDDS_MAX_DIMS = 8
GDDS_MAX_FIELDS = 256

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

# Keep numeric codes identical to include/gradients/tensor.h::gd_dtype.
DTYPE_CODES: dict[str, int] = {
    "f16": 1,
    "bf16": 2,
    "f32": 3,
    "i32": 4,
    "u8": 5,
}
DTYPE_NAMES = {v: k for k, v in DTYPE_CODES.items()}
DTYPE_SIZES: dict[str, int] = {
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "i32": 4,
    "u8": 1,
}

COLLATE_CODES: dict[str, int] = {
    "stack": 0,
    "pad_longest": 1,
    "packed_sequence": 2,
    "generated": 3,
}
GENERATED_CODES: dict[str, int] = {
    "none": 0,
    "lengths": 1,
    "mask": 2,
    "cu_seqlens": 3,
    "positions": 4,
}


@dataclass(frozen=True)
class FieldSpec:
    """One field in the GDDS schema.

    `shape` is the per-sample shape, without the dataloader batch dimension.
    Use -1 for the leading variable sequence dimension when `collate` is
    `pad_longest` or `packed_sequence`.

    Optional generated batch-only fields can be requested with `emit_lengths`,
    `emit_mask`, `emit_cu_seqlens`, and `emit_positions`. They are stored in the
    GDDS schema but not in per-sample records; the C dataloader materializes them
    during collation.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    collate: str = "stack"
    ragged_dim: int | None = None
    pad_value: Any = 0
    generated: str | None = None
    source: str | None = None
    emit_lengths: str | None = None
    emit_mask: str | None = None
    emit_cu_seqlens: str | None = None
    emit_positions: str | None = None


@dataclass(frozen=True)
class TensorData:
    """Pre-packed tensor payload for a single GDDS sample field."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (value + alignment - 1) // alignment * alignment


def _fnv64_update(h: int, data: bytes | bytearray | memoryview) -> int:
    for b in data:
        h ^= int(b)
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def _fnv64_u64(h: int, value: int) -> int:
    return _fnv64_update(h, struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))


def _fnv64_i64(h: int, value: int) -> int:
    return _fnv64_update(h, struct.pack("<q", int(value)))


def _fnv64_str(h: int, value: str) -> int:
    return _fnv64_update(h, value.encode("utf-8") + b"\0")


def _field_flags(field: FieldSpec, field_index: Mapping[str, int]) -> int:
    collate = COLLATE_CODES[field.collate]
    generated = GENERATED_CODES[field.generated or "none"]
    ragged = 0 if field.ragged_dim is None else int(field.ragged_dim) + 1
    source = 0 if field.source is None else field_index[field.source] + 1
    if not 0 <= ragged <= GDDS_MAX_DIMS:
        raise ValueError(f"field {field.name!r} ragged_dim is out of range")
    if not 0 <= source <= GDDS_MAX_FIELDS:
        raise ValueError(f"field {field.name!r} source index is out of range")
    return collate | (generated << 8) | (ragged << 16) | (source << 24)


def _pad_value_bits(field: FieldSpec) -> int:
    if field.collate != "pad_longest":
        return 0
    packed = pack_values(field.dtype, field.pad_value)
    if len(packed) != DTYPE_SIZES[field.dtype] or len(packed) > 8:
        raise ValueError(f"field {field.name!r} pad_value must be a scalar {field.dtype}")
    return int.from_bytes(packed.ljust(8, b"\0"), "little")


def schema_hash(fields: Sequence[FieldSpec | Mapping[str, Any]]) -> int:
    """Return the schema hash used by the C runtime."""

    fields_norm = _normalize_fields(fields)
    field_index = {field.name: i for i, field in enumerate(fields_norm)}
    h = FNV_OFFSET
    h = _fnv64_str(h, "gdds-schema-v1")
    h = _fnv64_u64(h, len(fields_norm))
    for field in fields_norm:
        h = _fnv64_str(h, field.name)
        h = _fnv64_u64(h, DTYPE_CODES[field.dtype])
        h = _fnv64_u64(h, len(field.shape))
        padded = tuple(field.shape) + (0,) * (GDDS_MAX_DIMS - len(field.shape))
        for dim in padded:
            h = _fnv64_i64(h, dim)
    metadata = any(_field_flags(field, field_index) or _pad_value_bits(field) for field in fields_norm)
    if metadata:
        h = _fnv64_str(h, "gdds-collate-v1")
        for field in fields_norm:
            h = _fnv64_u64(h, _field_flags(field, field_index))
            h = _fnv64_u64(h, _pad_value_bits(field))
    return h or 1


def _normalize_dtype(dtype: str) -> str:
    dtype = dtype.lower()
    if dtype not in DTYPE_CODES:
        raise ValueError(f"unsupported GDDS dtype: {dtype!r}")
    return dtype


def _normalize_collate(collate: str) -> str:
    collate = collate.lower()
    if collate not in COLLATE_CODES:
        raise ValueError(f"unsupported GDDS collate mode: {collate!r}")
    return collate


def _normalize_generated(generated: str | None) -> str | None:
    if generated is None:
        return None
    generated = generated.lower()
    if generated not in GENERATED_CODES or generated == "none":
        raise ValueError(f"unsupported GDDS generated field kind: {generated!r}")
    return generated


def _normalize_optional_name(value: Any) -> str | None:
    if value is None or value is False:
        return None
    if value is True:
        raise ValueError("generated field names must be explicit strings")
    text = str(value)
    if not text:
        raise ValueError("generated field name cannot be empty")
    return text


def _normalize_field(field: FieldSpec | Mapping[str, Any]) -> FieldSpec:
    if isinstance(field, FieldSpec):
        out = field
    else:
        out = FieldSpec(
            name=str(field["name"]),
            dtype=str(field["dtype"]),
            shape=tuple(int(x) for x in field["shape"]),
            collate=str(field.get("collate", "stack")),
            ragged_dim=None if field.get("ragged_dim") is None else int(field["ragged_dim"]),
            pad_value=field.get("pad_value", 0),
            generated=None if field.get("generated") is None else str(field["generated"]),
            source=None if field.get("source") is None else str(field["source"]),
            emit_lengths=_normalize_optional_name(field.get("emit_lengths")),
            emit_mask=_normalize_optional_name(field.get("emit_mask")),
            emit_cu_seqlens=_normalize_optional_name(field.get("emit_cu_seqlens")),
            emit_positions=_normalize_optional_name(field.get("emit_positions")),
        )
    dtype = _normalize_dtype(out.dtype)
    collate = _normalize_collate(out.collate)
    generated = _normalize_generated(out.generated)
    if not out.name:
        raise ValueError("field name cannot be empty")
    encoded = out.name.encode("utf-8")
    if len(encoded) >= GDDS_FIELD_NAME_MAX:
        raise ValueError(f"field name too long for GDDS: {out.name!r}")
    if len(out.shape) > GDDS_MAX_DIMS:
        raise ValueError(f"field {out.name!r} rank exceeds {GDDS_MAX_DIMS}")
    for dim in out.shape:
        if dim == -1:
            continue
        if dim <= 0:
            raise ValueError(f"field {out.name!r} has invalid shape {out.shape!r}")
    ragged_dim = out.ragged_dim
    if collate in {"pad_longest", "packed_sequence"}:
        variable_dims = [i for i, dim in enumerate(out.shape) if dim == -1]
        if variable_dims != [0]:
            raise ValueError(f"field {out.name!r} {collate} requires leading shape -1")
        if ragged_dim is None:
            ragged_dim = 0
        if ragged_dim != 0:
            raise ValueError(f"field {out.name!r} currently supports ragged_dim=0 only")
    elif collate == "stack":
        if any(dim == -1 for dim in out.shape):
            raise ValueError(f"field {out.name!r} stack collation requires fixed shape")
        ragged_dim = None
    elif collate == "generated":
        if generated is None or out.source is None:
            raise ValueError(f"generated field {out.name!r} requires generated and source")
        ragged_dim = None
    if collate != "generated" and (generated is not None or out.source is not None):
        raise ValueError(f"field {out.name!r} source/generated are only valid for generated fields")
    return FieldSpec(
        out.name,
        dtype,
        tuple(out.shape),
        collate,
        ragged_dim,
        out.pad_value,
        generated,
        out.source,
        _normalize_optional_name(out.emit_lengths),
        _normalize_optional_name(out.emit_mask),
        _normalize_optional_name(out.emit_cu_seqlens),
        _normalize_optional_name(out.emit_positions),
    )


def _generated_field(name: str, kind: str, source: FieldSpec) -> FieldSpec:
    dtype = "u8" if kind == "mask" else "i32"
    return FieldSpec(
        name=name,
        dtype=dtype,
        shape=(-1,),
        collate="generated",
        generated=kind,
        source=source.name,
    )


def _normalize_fields(fields: Sequence[FieldSpec | Mapping[str, Any]]) -> list[FieldSpec]:
    base = [_normalize_field(f) for f in fields]
    out: list[FieldSpec] = []
    for field in base:
        stored_field = FieldSpec(
            name=field.name,
            dtype=field.dtype,
            shape=field.shape,
            collate=field.collate,
            ragged_dim=field.ragged_dim,
            pad_value=field.pad_value,
            generated=field.generated,
            source=field.source,
        )
        out.append(stored_field)
        if field.collate == "generated":
            continue
        if field.emit_lengths is not None:
            out.append(_generated_field(field.emit_lengths, "lengths", field))
        if field.emit_mask is not None:
            if field.collate != "pad_longest":
                raise ValueError(f"field {field.name!r} emit_mask requires pad_longest collation")
            out.append(_generated_field(field.emit_mask, "mask", field))
        if field.emit_cu_seqlens is not None:
            if field.collate != "packed_sequence":
                raise ValueError(f"field {field.name!r} emit_cu_seqlens requires packed_sequence collation")
            out.append(_generated_field(field.emit_cu_seqlens, "cu_seqlens", field))
        if field.emit_positions is not None:
            if field.collate != "packed_sequence":
                raise ValueError(f"field {field.name!r} emit_positions requires packed_sequence collation")
            out.append(_generated_field(field.emit_positions, "positions", field))
    if not out:
        raise ValueError("GDDS schema must contain at least one field")
    if len(out) > GDDS_MAX_FIELDS:
        raise ValueError(f"GDDS supports at most {GDDS_MAX_FIELDS} fields")
    names = [f.name for f in out]
    if len(names) != len(set(names)):
        raise ValueError("GDDS field names must be unique")
    field_by_name = {field.name: field for field in out}
    for field in out:
        if field.source is None:
            continue
        source = field_by_name.get(field.source)
        if source is None:
            raise ValueError(f"generated field {field.name!r} references unknown source {field.source!r}")
        if source.collate == "generated":
            raise ValueError(f"generated field {field.name!r} cannot use generated source {field.source!r}")
        if field.generated == "lengths":
            if field.dtype != "i32" or source.collate not in {"pad_longest", "packed_sequence"}:
                raise ValueError(f"generated lengths field {field.name!r} requires i32 varlen source")
        elif field.generated == "mask":
            if field.dtype not in {"u8", "i32"} or source.collate != "pad_longest":
                raise ValueError(f"generated mask field {field.name!r} requires u8/i32 pad_longest source")
        elif field.generated == "cu_seqlens":
            if field.dtype != "i32" or source.collate != "packed_sequence":
                raise ValueError(f"generated cu_seqlens field {field.name!r} requires i32 packed_sequence source")
        elif field.generated == "positions":
            if field.dtype != "i32" or source.collate != "packed_sequence":
                raise ValueError(f"generated positions field {field.name!r} requires i32 packed_sequence source")
    return out


def _infer_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return ()
    if not isinstance(value, Sequence):
        return ()
    length = len(value)
    if length == 0:
        return (0,)
    child_shape = _infer_shape(value[0])
    for item in value[1:]:
        if _infer_shape(item) != child_shape:
            raise ValueError("ragged arrays must be packed explicitly as TensorData")
    return (length,) + child_shape


def _flatten(value: Any) -> Iterator[Any]:
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        yield value
        return
    if not isinstance(value, Sequence):
        yield value
        return
    for item in value:
        yield from _flatten(item)


def _numel(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    out = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"runtime tensor shape dimensions must be positive, got {shape!r}")
        out *= dim
    return out


def _pack_f16(values: Sequence[Any]) -> bytes:
    chunks: list[bytes] = []
    step = 8192
    for start in range(0, len(values), step):
        chunk = [float(x) for x in values[start : start + step]]
        chunks.append(struct.pack("<" + "e" * len(chunk), *chunk))
    return b"".join(chunks)


def _pack_f32(values: Sequence[Any]) -> bytes:
    chunks: list[bytes] = []
    step = 8192
    for start in range(0, len(values), step):
        chunk = [float(x) for x in values[start : start + step]]
        chunks.append(struct.pack("<" + "f" * len(chunk), *chunk))
    return b"".join(chunks)


def _pack_bf16(values: Sequence[Any]) -> bytes:
    out = bytearray(2 * len(values))
    for i, value in enumerate(values):
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        # Round to nearest-even before dropping the lower 16 mantissa bits.
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
        struct.pack_into("<H", out, 2 * i, (bits >> 16) & 0xFFFF)
    return bytes(out)


def _pack_i32(values: Sequence[Any]) -> bytes:
    chunks: list[bytes] = []
    step = 8192
    for start in range(0, len(values), step):
        chunk = [int(x) for x in values[start : start + step]]
        chunks.append(struct.pack("<" + "i" * len(chunk), *chunk))
    return b"".join(chunks)


def _pack_u8(values: Sequence[Any]) -> bytes:
    return bytes(int(x) & 0xFF for x in values)


def pack_values(dtype: str, values: Any) -> bytes:
    """Pack Python scalar/list data into GDDS little-endian bytes."""

    dtype = _normalize_dtype(dtype)
    flat = list(_flatten(values))
    if dtype == "f16":
        return _pack_f16(flat)
    if dtype == "bf16":
        return _pack_bf16(flat)
    if dtype == "f32":
        return _pack_f32(flat)
    if dtype == "i32":
        return _pack_i32(flat)
    if dtype == "u8":
        return _pack_u8(flat)
    raise AssertionError(dtype)


def tensor(value: Any, dtype: str, shape: Sequence[int] | None = None) -> TensorData:
    """Create a `TensorData` value for a sample field.

    `value` may be a scalar, nested Python lists/tuples, or raw bytes for `u8`.
    For raw bytes with non-u8 dtypes, pass `TensorData` directly so the shape is
    explicit and no accidental repacking occurs.
    """

    dtype = _normalize_dtype(dtype)
    if isinstance(value, TensorData):
        if value.dtype != dtype:
            raise ValueError(f"TensorData dtype {value.dtype!r} does not match {dtype!r}")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        if shape is None:
            if dtype != "u8":
                raise ValueError("raw non-u8 bytes require an explicit shape")
            shape_tuple = (len(value),)
        else:
            shape_tuple = tuple(int(x) for x in shape)
        data = bytes(value)
    else:
        shape_tuple = tuple(int(x) for x in (_infer_shape(value) if shape is None else shape))
        data = pack_values(dtype, value)
    expected = _numel(shape_tuple) * DTYPE_SIZES[dtype]
    if len(data) != expected:
        raise ValueError(
            f"packed tensor byte size mismatch for dtype={dtype}, shape={shape_tuple}: "
            f"got {len(data)}, expected {expected}"
        )
    return TensorData(dtype=dtype, shape=shape_tuple, data=data)


def _sample_tensor(value: Any, field: FieldSpec) -> TensorData:
    data = value if isinstance(value, TensorData) else tensor(value, field.dtype)
    if data.dtype != field.dtype:
        raise ValueError(f"sample field {field.name!r} has dtype {data.dtype!r}, expected {field.dtype!r}")
    if len(data.shape) != len(field.shape):
        raise ValueError(f"sample field {field.name!r} has rank {len(data.shape)}, expected {len(field.shape)}")
    for actual, expected in zip(data.shape, field.shape):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"sample field {field.name!r} has shape {data.shape!r}, expected schema shape {field.shape!r}"
            )
    return data


def _encode_schema(fields: Sequence[FieldSpec]) -> bytes:
    field_index = {field.name: i for i, field in enumerate(fields)}
    out = bytearray(len(fields) * GDDS_FIELD_DESC_SIZE)
    for i, field in enumerate(fields):
        base = i * GDDS_FIELD_DESC_SIZE
        name = field.name.encode("utf-8")
        out[base : base + len(name)] = name
        struct.pack_into("<I", out, base + 64, DTYPE_CODES[field.dtype])
        struct.pack_into("<I", out, base + 68, len(field.shape))
        padded = tuple(field.shape) + (0,) * (GDDS_MAX_DIMS - len(field.shape))
        struct.pack_into("<8q", out, base + 72, *padded)
        struct.pack_into("<Q", out, base + 136, _field_flags(field, field_index))
        struct.pack_into("<Q", out, base + 144, _pad_value_bits(field))
    return bytes(out)


def _encode_record(fields: Sequence[FieldSpec], sample: Mapping[str, Any]) -> bytes:
    payload = bytearray()
    entries: list[bytes] = []
    for field_id, field in enumerate(fields):
        if field.collate == "generated":
            continue
        if field.name not in sample:
            raise ValueError(f"sample missing GDDS field {field.name!r}")
        value = _sample_tensor(sample[field.name], field)
        offset = len(payload)
        payload.extend(value.data)
        padded_shape = tuple(value.shape) + (0,) * (GDDS_MAX_DIMS - len(value.shape))
        entries.append(
            struct.pack(
                "<HHI8qQQ",
                field_id,
                len(value.shape),
                0,
                *padded_shape,
                offset,
                len(value.data),
            )
        )
    header_nbytes = GDDS_RECORD_HEADER_SIZE + len(entries) * GDDS_RECORD_FIELD_DESC_SIZE
    return b"".join(
        [
            struct.pack("<4sHHIQ", GDDS_RECORD_MAGIC, len(entries), 0, header_nbytes, len(payload)),
            *entries,
            bytes(payload),
        ]
    )


def _build_header(
    *,
    field_count: int,
    n_samples: int,
    schema_offset: int,
    index_offset: int,
    data_offset: int,
    data_nbytes: int,
    schema_hash_value: int,
) -> bytes:
    header = bytearray(GDDS_HEADER_SIZE)
    header[0:8] = GDDS_MAGIC
    struct.pack_into("<I", header, 8, GDDS_VERSION)
    struct.pack_into("<I", header, 12, GDDS_HEADER_SIZE)
    struct.pack_into("<I", header, 16, field_count)
    struct.pack_into("<I", header, 20, 0)
    struct.pack_into("<Q", header, 24, n_samples)
    struct.pack_into("<Q", header, 32, schema_offset)
    struct.pack_into("<Q", header, 40, index_offset)
    struct.pack_into("<Q", header, 48, data_offset)
    struct.pack_into("<Q", header, 56, data_nbytes)
    struct.pack_into("<Q", header, 64, schema_hash_value)
    return bytes(header)


def field_metadata(field: FieldSpec) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": field.name,
        "dtype": field.dtype,
        "shape": list(field.shape),
        "collate": field.collate,
    }
    if field.ragged_dim is not None:
        item["ragged_dim"] = field.ragged_dim
    if field.collate == "pad_longest":
        item["pad_value"] = field.pad_value
    if field.collate == "generated":
        item["generated"] = field.generated
        item["source"] = field.source
    return item


def write_gdds_shard(
    path: str | Path,
    fields: Sequence[FieldSpec | Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
) -> Path:
    """Write one GDDS shard and return its path.

    Records are streamed to disk; memory usage is O(number_of_samples) for the
    16-byte-per-sample index plus one encoded record at a time.
    """

    normalized_fields = _normalize_fields(fields)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    schema = _encode_schema(normalized_fields)
    n_samples = len(samples)
    schema_offset = GDDS_HEADER_SIZE
    index_offset = align_up(schema_offset + len(schema), 64)
    index_nbytes = n_samples * GDDS_INDEX_ENTRY_SIZE
    data_offset = align_up(index_offset + index_nbytes, 64)
    schema_hash_value = schema_hash(normalized_fields)
    index: list[tuple[int, int]] = []
    cursor = data_offset
    with tmp.open("wb+") as f:
        f.write(b"\0" * GDDS_HEADER_SIZE)
        f.write(schema)
        f.write(b"\0" * (index_offset - f.tell()))
        f.write(b"\0" * index_nbytes)
        f.write(b"\0" * (data_offset - f.tell()))
        for sample in samples:
            record = _encode_record(normalized_fields, sample)
            index.append((cursor, len(record)))
            f.write(record)
            cursor += len(record)
        data_nbytes = cursor - data_offset
        f.seek(index_offset)
        for offset, nbytes in index:
            f.write(struct.pack("<QQ", offset, nbytes))
        f.seek(0)
        f.write(
            _build_header(
                field_count=len(normalized_fields),
                n_samples=n_samples,
                schema_offset=schema_offset,
                index_offset=index_offset,
                data_offset=data_offset,
                data_nbytes=data_nbytes,
                schema_hash_value=schema_hash_value,
            )
        )
        f.truncate(cursor)
        f.flush()
    os.replace(tmp, path)
    return path


def _chunks(items: Iterable[Mapping[str, Any]], size: int) -> Iterator[list[Mapping[str, Any]]]:
    chunk: list[Mapping[str, Any]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_gdds_split(
    directory: str | Path,
    split: str,
    fields: Sequence[FieldSpec | Mapping[str, Any]],
    samples: Iterable[Mapping[str, Any]],
    *,
    samples_per_shard: int = 8192,
    write_manifest: bool = True,
) -> list[Path]:
    """Write a sharded GDDS split named `<split>-00000.gdds`, ..."""

    if not split:
        raise ValueError("split name cannot be empty")
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    fields_norm = _normalize_fields(fields)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = 0
    for shard_idx, chunk in enumerate(_chunks(samples, samples_per_shard)):
        path = directory / f"{split}-{shard_idx:05d}.gdds"
        paths.append(write_gdds_shard(path, fields_norm, chunk))
        total += len(chunk)
    if not paths:
        raise ValueError("cannot write an empty GDDS split")
    if write_manifest:
        manifest = {
            "format": "GDDS",
            "version": GDDS_VERSION,
            "schema_hash": f"0x{schema_hash(fields_norm):016x}",
            "fields": [field_metadata(f) for f in fields_norm],
            "splits": {
                split: {
                    "samples": total,
                    "shards": [p.name for p in paths],
                }
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return paths


__all__ = [
    "DTYPE_CODES",
    "DTYPE_NAMES",
    "DTYPE_SIZES",
    "FieldSpec",
    "field_metadata",
    "TensorData",
    "pack_values",
    "schema_hash",
    "tensor",
    "write_gdds_shard",
    "write_gdds_split",
]
