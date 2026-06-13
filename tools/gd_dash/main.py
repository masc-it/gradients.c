#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.115",
#   "jinja2>=3.1",
#   "uvicorn[standard]>=0.30",
# ]
# ///
"""Live metrics dashboard for gradients.c JSONL training logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

METRIC_PRIORITY = {
    "loss": 0,
    "val_loss": 1,
    "best_val_loss": 2,
    "lr": 10,
    "tok_s": 11,
    "batch_s": 12,
}

CONFIG_NUMERIC_KEYS = {
    "v",
    "time",
    "step",
    "epoch",
    "epoch_step",
    "steps_per_epoch",
    "total_steps",
    "samples_per_epoch",
    "dataset_samples",
    "dataset_tokens",
    "val_samples",
    "val_tokens",
    "total_params",
    "trainable_params",
    "param_bytes",
    "memory_params_bytes",
    "memory_state_bytes",
    "memory_scratch_slot_bytes",
    "memory_data_slot_bytes",
    "vocab_size",
    "context_length",
    "d_model",
    "heads",
    "head_dim",
    "mlp_hidden",
    "layers",
    "epochs",
    "batch_size",
    "report_every",
    "early_stopping_patience",
    "warmup_steps",
    "seed",
    "dropout",
    "lr_max",
    "lr_min",
    "weight_decay",
    "grad_clip_norm",
    "optimizer_steps",
    "patience",
    "params_watermark_bytes",
    "state_watermark_bytes",
    "scratch_max_slot_watermark_bytes",
    "data_max_slot_watermark_bytes",
    "backend_waits",
}


def is_safe_segment(value: str) -> bool:
    return bool(value and SAFE_SEGMENT.fullmatch(value) and value not in {".", ".."})


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_chart_metric(key: str, value: Any) -> bool:
    return is_number(value) and key not in CONFIG_NUMERIC_KEYS


def metric_sort_key(name: str) -> tuple[int, str]:
    return (METRIC_PRIORITY.get(name, 100), name)


def read_last_complete_line(path: Path, chunk_size: int = 8192) -> bytes | None:
    size = path.stat().st_size
    if size == 0:
        return None
    with path.open("rb") as handle:
        position = size
        suffix = b""
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            data = handle.read(read_size) + suffix
            lines = data.splitlines()
            if data.endswith((b"\n", b"\r")):
                if lines:
                    return lines[-1]
            elif len(lines) >= 2:
                return lines[-2]
            suffix = data
    lines = suffix.splitlines()
    return lines[-1] if lines else None


@dataclass(frozen=True)
class RunInfo:
    project: str
    run_id: str
    path: Path
    size: int
    mtime: float
    last_event: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "run_id": self.run_id,
            "size": self.size,
            "mtime": self.mtime,
            "path": str(self.path),
            "last_event": self.last_event,
        }


class MetricsStore:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    def list_runs(self) -> list[RunInfo]:
        if not self.root.exists():
            return []
        runs: list[RunInfo] = []
        for project_dir in sorted(self.root.iterdir()):
            if not project_dir.is_dir() or not is_safe_segment(project_dir.name):
                continue
            for path in sorted(project_dir.glob("*.jsonl")):
                run_id = path.stem
                if not is_safe_segment(run_id):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                last_event = self._read_last_event(path)
                runs.append(
                    RunInfo(
                        project=project_dir.name,
                        run_id=run_id,
                        path=path,
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        last_event=last_event,
                    )
                )
        runs.sort(key=lambda run: (run.mtime, run.project, run.run_id), reverse=True)
        return runs

    def get_run(self, project: str, run_id: str) -> RunInfo:
        path = self.resolve(project, run_id)
        try:
            stat = path.stat()
        except OSError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return RunInfo(
            project=project,
            run_id=run_id,
            path=path,
            size=stat.st_size,
            mtime=stat.st_mtime,
            last_event=self._read_last_event(path),
        )

    def resolve(self, project: str, run_id: str) -> Path:
        if not is_safe_segment(project) or not is_safe_segment(run_id):
            raise HTTPException(status_code=404, detail="run not found")
        path = (self.root / project / f"{run_id}.jsonl").resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="run not found")
        return path

    def _read_last_event(self, path: Path) -> dict[str, Any] | None:
        try:
            line = read_last_complete_line(path)
        except OSError:
            return None
        if not line:
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return None
        return event if isinstance(event, dict) else None

    def read_complete_events(self, path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int, bool]:
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        if offset < 0:
            raise HTTPException(status_code=400, detail="offset must be non-negative")
        if offset > size:
            return [], 0, True

        events: list[dict[str, Any]] = []
        complete_offset = offset
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                while True:
                    line_offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        complete_offset = line_offset
                        break
                    complete_offset = handle.tell()
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        event = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict):
                        events.append(event)
        except OSError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return events, complete_offset, False


def downsample_points(points: list[list[float | str]], max_points: int) -> list[list[float | str]]:
    if max_points <= 0 or len(points) <= max_points:
        return points
    if max_points <= 2:
        return [points[0], points[-1]]

    target_buckets = max(1, (max_points - 2) // 2)
    bucket_size = max(1, math.ceil((len(points) - 2) / target_buckets))
    sampled: list[list[float | str]] = [points[0]]
    end_limit = len(points) - 1
    start = 1
    while start < end_limit:
        bucket = points[start : min(start + bucket_size, end_limit)]
        min_point = min(bucket, key=lambda point: float(point[1]))
        max_point = max(bucket, key=lambda point: float(point[1]))
        if min_point[0] <= max_point[0]:
            candidates = [min_point, max_point]
        else:
            candidates = [max_point, min_point]
        for candidate in candidates:
            if sampled[-1] is not candidate:
                sampled.append(candidate)
        start += bucket_size
    if sampled[-1] is not points[-1]:
        sampled.append(points[-1])
    if len(sampled) <= max_points:
        return sampled
    stride = math.ceil(len(sampled) / max_points)
    thinned = sampled[::stride]
    if thinned[-1] is not sampled[-1]:
        thinned.append(sampled[-1])
    return thinned


def build_series(events: list[dict[str, Any]], max_points: int = 3000) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[list[float | str]]] = defaultdict(list)
    axes: dict[str, str] = {}
    for event in events:
        x_axis = "step" if is_number(event.get("step")) else "time"
        x_value = event.get("step") if x_axis == "step" else event.get("time")
        if not is_number(x_value):
            continue
        event_name = str(event.get("event", ""))
        event_time = float(event.get("time")) if is_number(event.get("time")) else float(x_value)
        for key, value in event.items():
            if not is_chart_metric(key, value):
                continue
            grouped[key].append([float(x_value), float(value), event_time, event_name])
            axes.setdefault(key, x_axis)
    return {
        key: {"axis": axes.get(key, "step"), "points": downsample_points(points, max_points)}
        for key, points in sorted(grouped.items(), key=lambda item: metric_sort_key(item[0]))
    }


def latest_values(events: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in reversed(events):
        for key, value in event.items():
            if key in latest or not is_chart_metric(key, value):
                continue
            latest[key] = {
                "name": key,
                "value": float(value),
                "step": event.get("step"),
                "time": event.get("time"),
                "event": event.get("event"),
            }
    return sorted(latest.values(), key=lambda item: metric_sort_key(str(item["name"])))[:limit]


def choose_selected(runs: list[RunInfo], project: str | None, run_id: str | None) -> RunInfo | None:
    if project and run_id:
        for run in runs:
            if run.project == project and run.run_id == run_id:
                return run
    return runs[0] if runs else None


def create_app(metrics_dir: Path) -> FastAPI:
    app = FastAPI(title="gradients.c metrics dashboard")
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    store = MetricsStore(metrics_dir)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request, project: str | None = None, run: str | None = None) -> HTMLResponse:
        runs = store.list_runs()
        selected = choose_selected(runs, project, run)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "metrics_root": str(store.root),
                "runs": runs,
                "selected": selected,
            },
        )

    @app.get("/partials/runs", response_class=HTMLResponse)
    async def partial_runs(request: Request, project: str | None = None, run: str | None = None) -> HTMLResponse:
        runs = store.list_runs()
        selected = choose_selected(runs, project, run)
        return templates.TemplateResponse(
            request,
            "components/run_list.html",
            {"runs": runs, "selected": selected},
        )

    @app.get("/api/runs")
    async def api_runs() -> dict[str, Any]:
        return {"runs": [run.as_dict() for run in store.list_runs()]}

    @app.get("/api/runs/{project}/{run_id}/snapshot")
    async def api_snapshot(
        project: str,
        run_id: str,
        max_points: int = Query(default=3000, ge=100, le=20000),
    ) -> dict[str, Any]:
        run = store.get_run(project, run_id)
        events, offset, reset = store.read_complete_events(run.path, 0)
        return {
            "run": run.as_dict(),
            "offset": offset,
            "reset": reset,
            "event_count": len(events),
            "series": build_series(events, max_points=max_points),
            "latest_values": latest_values(events),
        }

    @app.get("/api/runs/{project}/{run_id}/tail")
    async def api_tail(
        project: str,
        run_id: str,
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        run = store.get_run(project, run_id)
        events, next_offset, reset = store.read_complete_events(run.path, offset)
        if reset:
            return {"run": run.as_dict(), "offset": next_offset, "reset": True, "series": {}}
        return {
            "run": run.as_dict(),
            "offset": next_offset,
            "reset": False,
            "event_count": len(events),
            "series": build_series(events, max_points=0),
            "latest_values": latest_values(events),
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live dashboard for gradients.c metrics JSONL logs.")
    parser.add_argument("--metrics-dir", default="data/metrics", help="metrics root directory")
    parser.add_argument("--host", default="127.0.0.1", help="bind host")
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(Path(args.metrics_dir))
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
