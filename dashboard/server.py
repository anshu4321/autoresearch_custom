#!/usr/bin/env python3
"""Live dashboard server for autoresearch retrieval experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass
class ServerConfig:
    repo_root: Path
    static_dir: Path
    results_path: Path
    run_log_path: Path


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            val = _safe_float(row.get("val_ndcg_at_10", "0"))
            mem = _safe_float(row.get("memory_gb", "0"))
            status = (row.get("status") or "").strip().lower() or "unknown"
            description = (row.get("description") or "").strip()
            commit = (row.get("commit") or "").strip()
            rows.append(
                {
                    "index": idx,
                    "commit": commit,
                    "val_ndcg_at_10": val,
                    "memory_gb": mem,
                    "status": status,
                    "description": description,
                }
            )
    return rows


def _extract_metric(text: str, key: str) -> float | None:
    m = re.search(rf"^{re.escape(key)}:\s+([0-9]+(?:\.[0-9]+)?)$", text, flags=re.MULTILINE)
    if not m:
        return None
    return _safe_float(m.group(1), default=0.0)


def _read_log_state(run_log_path: Path) -> dict[str, Any]:
    if not run_log_path.exists():
        return {
            "state": "missing",
            "progress": None,
            "summary": None,
            "age_seconds": None,
            "updated_at": None,
        }

    text = run_log_path.read_text(encoding="utf-8", errors="replace")
    normalized = text.replace("\r", "\n")
    mtime = run_log_path.stat().st_mtime
    age = max(0.0, time.time() - mtime)

    step_re = re.compile(
        r"step\s+(\d+)\s+\(([0-9]+(?:\.[0-9]+)?)%\)\s+\|\s+"
        r"loss:\s+([0-9]+(?:\.[0-9]+)?)\s+\|\s+dt:\s+(\d+)ms\s+\|\s+remaining:\s+(\d+)s"
    )
    steps = step_re.findall(normalized)
    progress = None
    if steps:
        s = steps[-1]
        progress = {
            "step": int(s[0]),
            "pct": _safe_float(s[1]),
            "loss": _safe_float(s[2]),
            "dt_ms": int(s[3]),
            "remaining_s": int(s[4]),
        }

    summary = {
        "val_ndcg_at_10": _extract_metric(normalized, "val_ndcg_at_10"),
        "val_recall_at_10": _extract_metric(normalized, "val_recall_at_10"),
        "training_seconds": _extract_metric(normalized, "training_seconds"),
        "total_seconds": _extract_metric(normalized, "total_seconds"),
        "peak_vram_mb": _extract_metric(normalized, "peak_vram_mb"),
        "num_steps": _extract_metric(normalized, "num_steps"),
    }
    has_summary = summary["val_ndcg_at_10"] is not None

    if progress and not has_summary and age <= 30:
        state = "running"
    elif progress and not has_summary:
        state = "stalled"
    elif has_summary:
        state = "complete"
    else:
        state = "idle"

    return {
        "state": state,
        "progress": progress,
        "summary": summary,
        "age_seconds": age,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
    }


def _read_git_info(repo_root: Path) -> dict[str, str]:
    def _run(cmd: list[str]) -> str:
        try:
            out = subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    return {
        "branch": _run(["git", "branch", "--show-current"]),
        "head": _run(["git", "rev-parse", "--short", "HEAD"]),
    }


def _build_snapshot(config: ServerConfig) -> dict[str, Any]:
    rows = _read_results(config.results_path)
    log_state = _read_log_state(config.run_log_path)
    git_info = _read_git_info(config.repo_root)

    best_row = max(rows, key=lambda r: r["val_ndcg_at_10"], default=None)
    latest_row = rows[-1] if rows else None
    keeps = sum(1 for r in rows if r["status"] == "keep")
    discards = sum(1 for r in rows if r["status"] == "discard")
    crashes = sum(1 for r in rows if r["status"] == "crash")

    best_progress = []
    running_best = -math.inf
    for r in rows:
        running_best = max(running_best, r["val_ndcg_at_10"])
        best_progress.append(running_best)

    training_seconds = 300.0
    summary = log_state.get("summary") or {}
    if summary.get("training_seconds"):
        training_seconds = max(1.0, float(summary["training_seconds"]))

    cycle_seconds = max(315.0, training_seconds + 8.0)
    projected_runs_7h = int((7 * 3600) // cycle_seconds)

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "git": git_info,
        "stats": {
            "total_runs": len(rows),
            "keeps": keeps,
            "discards": discards,
            "crashes": crashes,
            "keep_ratio": (keeps / len(rows)) if rows else 0.0,
            "best": best_row,
            "latest": latest_row,
            "projected_runs_7h": projected_runs_7h,
            "cycle_seconds": cycle_seconds,
        },
        "series": {
            "labels": [str(r["index"]) for r in rows],
            "ndcg": [r["val_ndcg_at_10"] for r in rows],
            "best_so_far": best_progress,
        },
        "runs": rows,
        "live": log_state,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "AutoresearchDashboard/1.0"

    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any):
        super().__init__(*args, directory=directory, **kwargs)

    @property
    def app_config(self) -> ServerConfig:
        return self.server.app_config  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)

        if parsed.path == "/api/snapshot":
            payload = _build_snapshot(self.app_config)
            body = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/health":
            body = b'{"ok":true}'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/":
            self.path = "/index.html"

        return super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoresearch dashboard server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8787, help="Port to bind")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument(
        "--results",
        default="results_retrieval.tsv",
        help="Path to results TSV, relative to repo root",
    )
    parser.add_argument(
        "--run-log",
        default="run_retrieval.log",
        help="Path to live run log, relative to repo root",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    static_dir = (repo_root / "dashboard").resolve()
    config = ServerConfig(
        repo_root=repo_root,
        static_dir=static_dir,
        results_path=(repo_root / args.results).resolve(),
        run_log_path=(repo_root / args.run_log).resolve(),
    )

    if not config.static_dir.exists():
        raise SystemExit(f"Missing static directory: {config.static_dir}")

    handler_cls = lambda *h_args, **h_kwargs: DashboardHandler(  # noqa: E731
        *h_args,
        directory=str(config.static_dir),
        **h_kwargs,
    )

    httpd = ThreadingHTTPServer((args.host, args.port), handler_cls)
    httpd.app_config = config  # type: ignore[attr-defined]

    print(f"Dashboard server running on http://{args.host}:{args.port}")
    print(f"Repo root: {repo_root}")
    print(f"Results:   {config.results_path}")
    print(f"Run log:   {config.run_log_path}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
