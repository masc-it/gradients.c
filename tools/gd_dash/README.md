# gd_dash

Live HTMX/Tailwind dashboard for gradients.c metrics JSONL logs.

```sh
uv run tools/gd_dash/main.py --metrics-dir data/metrics --port 8765
```

Training runs write logs under:

```text
data/metrics/<project>/<run_id>.jsonl
```

The dashboard parses the full file once for the initial snapshot, then polls only appended bytes using the last complete JSONL byte offset.
