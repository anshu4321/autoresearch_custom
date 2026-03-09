#!/usr/bin/env python3
"""Autonomous experiment loop for retrieval autoresearch."""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


RESULTS_HEADER = ["commit", "val_ndcg_at_10", "memory_gb", "status", "description"]


@dataclass
class RunOutcome:
    status: Literal["keep", "discard", "crash"]
    val_ndcg: float
    peak_vram_mb: float
    description: str
    commit: str


def run_cmd(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=check,
    )


def git_short_head(repo: Path) -> str:
    return run_cmd(["git", "rev-parse", "--short", "HEAD"], repo).stdout.strip()


def ensure_results_file(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(RESULTS_HEADER)


def load_best_ndcg(path: Path) -> float:
    if not path.exists():
        return float("-inf")

    best = float("-inf")
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                value = float(row.get("val_ndcg_at_10", ""))
            except ValueError:
                continue
            if value > best:
                best = value
    return best


def append_result(path: Path, outcome: RunOutcome) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                outcome.commit,
                f"{outcome.val_ndcg:.6f}",
                f"{(outcome.peak_vram_mb / 1024.0):.1f}",
                outcome.status,
                outcome.description,
            ]
        )


def log_line(message: str) -> None:
    print(message, flush=True)


def parse_metrics(run_log_path: Path) -> tuple[float | None, float | None]:
    text = run_log_path.read_text(encoding="utf-8", errors="replace") if run_log_path.exists() else ""

    ndcg_match = re.search(r"^val_ndcg_at_10:\s+([0-9]+(?:\.[0-9]+)?)$", text, flags=re.MULTILINE)
    vram_match = re.search(r"^peak_vram_mb:\s+([0-9]+(?:\.[0-9]+)?)$", text, flags=re.MULTILINE)

    ndcg = float(ndcg_match.group(1)) if ndcg_match else None
    vram = float(vram_match.group(1)) if vram_match else None
    return ndcg, vram


def apply_mutation(file_path: Path, mutation_key: str, new_literal: str) -> tuple[str, str]:
    text = file_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^(\s*{re.escape(mutation_key)}\s*=\s*)([^#\n]+)(.*)$", flags=re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise RuntimeError(f"Could not find mutable key: {mutation_key}")

    old_literal = m.group(2).strip()
    replaced = pattern.sub(rf"\g<1>{new_literal}\g<3>", text, count=1)
    file_path.write_text(replaced, encoding="utf-8")
    return old_literal, new_literal


def choose_mutation(rng: random.Random, current_text: str) -> tuple[str, str, str]:
    space = {
        "TRAIN_BATCH_SIZE": ["128", "192", "256", "320", "384", "448"],
        "LEARNING_RATE": ["1e-5", "1.5e-5", "2e-5", "3e-5", "4e-5", "5e-5"],
        "UNFREEZE_LAST_N": ["1", "2", "3", "4", "5", "-1"],
        "PROJECTION_DIM": ["128", "256", "384", "512"],
        "WEIGHT_DECAY": ["0.0", "0.01", "0.03", "0.05"],
        "TEMPERATURE": ["0.03", "0.05", "0.07", "0.1"],
        "MAX_DOC_LEN": ["192", "256", "320"],
        "MAX_QUERY_LEN": ["32", "48", "64"],
    }

    key = rng.choice(list(space.keys()))
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^#\n]+)", flags=re.MULTILINE)
    m = pattern.search(current_text)
    if not m:
        raise RuntimeError(f"Could not parse current value for {key}")
    current = m.group(1).strip()

    values = space[key]
    if current not in values:
        target = rng.choice(values)
    else:
        idx = values.index(current)
        neighbors = []
        if idx - 1 >= 0:
            neighbors.append(values[idx - 1])
        if idx + 1 < len(values):
            neighbors.append(values[idx + 1])
        if rng.random() < 0.7 and neighbors:
            target = rng.choice(neighbors)
        else:
            target = rng.choice([v for v in values if v != current])

    return key, current, target


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autonomous retrieval experiment loop")
    parser.add_argument("--repo", default=".", help="Repo root")
    parser.add_argument("--hours", type=float, default=7.0, help="Loop duration in hours")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional hard cap on runs (0 = unlimited)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--improve-eps", type=float, default=1e-6)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    train_path = repo / "train_retrieval.py"
    run_log_path = repo / "run_retrieval.log"
    results_path = repo / "results_retrieval.tsv"

    if not train_path.exists():
        raise SystemExit(f"Missing file: {train_path}")

    ensure_results_file(results_path)
    best_ndcg = load_best_ndcg(results_path)

    rng = random.Random(args.seed)
    start = time.time()
    deadline = start + args.hours * 3600
    uv_bin = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")

    if not Path(uv_bin).exists():
        raise SystemExit(f"Could not locate uv executable. Checked: {uv_bin}")

    run_idx = 0
    keep_count = 0

    log_line(f"[loop] repo={repo}")
    log_line(f"[loop] starting best nDCG@10={best_ndcg:.6f}" if best_ndcg > -1e8 else "[loop] no prior best")
    log_line(f"[loop] deadline in {args.hours:.2f}h")

    while time.time() < deadline:
        if args.max_runs > 0 and run_idx >= args.max_runs:
            break

        run_idx += 1
        head_before = git_short_head(repo)
        current_text = train_path.read_text(encoding="utf-8")
        key, old_val, new_val = choose_mutation(rng, current_text)

        apply_mutation(train_path, key, new_val)
        desc = f"{key} {old_val} -> {new_val}"

        log_line(f"[run {run_idx:03d}] start | base={head_before} | {desc}")
        with run_log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.run(
                ["timeout", "900", uv_bin, "run", "train_retrieval.py"],
                cwd=str(repo),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        ndcg, peak_vram = parse_metrics(run_log_path)
        if ndcg is None:
            train_path.write_text(current_text, encoding="utf-8")
            fail_hint = "timeout" if proc.returncode == 124 else f"exit={proc.returncode}"
            outcome = RunOutcome(
                status="crash",
                val_ndcg=0.0,
                peak_vram_mb=0.0,
                description=f"{desc} ({fail_hint})",
                commit=f"wip{run_idx:04d}",
            )
            append_result(results_path, outcome)
            log_line(f"[run {run_idx:03d}] crash | {desc} | {fail_hint}")
            time.sleep(1.0)
            continue

        if peak_vram is None:
            peak_vram = 0.0

        if ndcg > best_ndcg + args.improve_eps:
            run_cmd(["git", "add", "train_retrieval.py"], repo)
            msg = f"retrieval auto: {desc} | ndcg {ndcg:.6f}"
            run_cmd(["git", "commit", "-m", msg], repo)
            commit = git_short_head(repo)
            best_ndcg = ndcg
            keep_count += 1
            outcome = RunOutcome(
                status="keep",
                val_ndcg=ndcg,
                peak_vram_mb=peak_vram,
                description=desc,
                commit=commit,
            )
            append_result(results_path, outcome)
            log_line(f"[run {run_idx:03d}] keep    | ndcg={ndcg:.6f} | vram={peak_vram/1024:.1f} GB | commit={commit}")
        else:
            train_path.write_text(current_text, encoding="utf-8")
            outcome = RunOutcome(
                status="discard",
                val_ndcg=ndcg,
                peak_vram_mb=peak_vram,
                description=desc,
                commit=f"wip{run_idx:04d}",
            )
            append_result(results_path, outcome)
            log_line(
                f"[run {run_idx:03d}] discard | ndcg={ndcg:.6f} (best={best_ndcg:.6f}) "
                f"| vram={peak_vram/1024:.1f} GB"
            )

    elapsed_h = (time.time() - start) / 3600.0
    log_line(f"[loop] finished runs={run_idx} keeps={keep_count} elapsed={elapsed_h:.2f}h best={best_ndcg:.6f}")


if __name__ == "__main__":
    main()
