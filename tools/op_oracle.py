"""Shared helpers for live operation oracle scripts.

The per-op `src/ops/*/{fwd,bwd}.py` scripts own their test-case generation,
tensor serialization, and numerical comparisons.  This module centralizes the
repo/build/compiler plumbing that was previously copied into each script.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


def repo_root(script_file: str | os.PathLike[str]) -> Path:
    """Return the gradients.c repository root for a script under src/ops/<op>."""

    return Path(script_file).resolve().parents[3]


def build_library(root: Path) -> None:
    """Build libgradients.a and the Metal library when supported by the host."""

    subprocess.run(["make", "build"], cwd=root, check=True)


def gradients_env(root: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return an environment with GRADIENTS_METALLIB pointing at the build output."""

    env = dict(os.environ if base is None else base)
    env["GRADIENTS_METALLIB"] = str(root / "build" / "gradients.metallib")
    return env


def compile_runner(
    root: Path,
    tmp: Path,
    name: str,
    source_text: str,
    *,
    extra_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
) -> Path:
    """Compile a temporary C oracle runner linked against build/libgradients.a.

    `source_text` may include `tools/oracle_runner_common.c`; the repository root
    is added to the include path alongside the public headers.
    """

    source = tmp / f"{name}.c"
    binary = tmp / name
    source.write_text(source_text)

    cmd = [
        "cc",
        f"-I{root / 'include'}",
        f"-I{root}",
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        *extra_cflags,
        str(source),
        str(root / "build" / "libgradients.a"),
        "-pthread",
        *extra_ldflags,
    ]
    if platform.system() == "Darwin":
        cmd.extend(["-framework", "Foundation", "-framework", "Metal"])
    cmd.extend(["-o", str(binary)])
    subprocess.run(cmd, cwd=root, check=True)
    return binary
