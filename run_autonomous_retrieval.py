#!/usr/bin/env python3
"""Autonomous experiment loop for retrieval autoresearch."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


RESULTS_HEADER = ["commit", "val_ndcg_at_10", "memory_gb", "status", "description"]

MUTATION_SPACE: dict[str, list[str]] = {
    "TRAIN_BATCH_SIZE": ["128", "192", "256", "320", "384", "448"],
    "LEARNING_RATE": ["1e-5", "1.5e-5", "2e-5", "3e-5", "4e-5", "5e-5"],
    "UNFREEZE_LAST_N": ["1", "2", "3", "4", "5", "-1"],
    "PROJECTION_DIM": ["128", "256", "384", "512"],
    "WEIGHT_DECAY": ["0.0", "0.01", "0.03", "0.05"],
    "TEMPERATURE": ["0.03", "0.05", "0.07", "0.1"],
    "MAX_DOC_LEN": ["192", "256", "320"],
    "MAX_QUERY_LEN": ["32", "48", "64"],
}

MUTATION_SEGMENT = re.compile(r"^([A-Z_]+)\s+(.+?)\s*->\s*(.+)$")


@dataclass
class RunOutcome:
    status: Literal["keep", "discard", "crash"]
    val_ndcg: float
    peak_vram_mb: float
    description: str
    commit: str


@dataclass
class ArmStats:
    trials: int = 0
    keeps: int = 0
    delta_sum: float = 0.0
    last_used_run: int = 0


class TabuMemory:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.queue: deque[str] = deque()
        self.counts: dict[str, int] = {}

    def add(self, signature: str) -> None:
        self.queue.append(signature)
        self.counts[signature] = self.counts.get(signature, 0) + 1

        while len(self.queue) > self.max_size:
            old = self.queue.popleft()
            n = self.counts.get(old, 0)
            if n <= 1:
                self.counts.pop(old, None)
            else:
                self.counts[old] = n - 1

    def contains(self, signature: str) -> bool:
        return signature in self.counts


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


def mutation_signature(base_commit: str, key: str, new_literal: str) -> str:
    return f"{base_commit}|{key}|{new_literal}"


def parse_current_literal(text: str, key: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*([^#\n]+)", flags=re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise RuntimeError(f"Could not parse current value for {key}")
    return m.group(1).strip()


def replace_literal_in_text(text: str, key: str, new_literal: str) -> tuple[str, str]:
    pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#\n]+)(.*)$", flags=re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise RuntimeError(f"Could not find mutable key: {key}")

    old_literal = m.group(2).strip()
    replaced = pattern.sub(rf"\g<1>{new_literal}\g<3>", text, count=1)
    return replaced, old_literal


def parse_mutations_from_description(description: str) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for seg in description.split(";"):
        part = seg.strip()
        if not part:
            continue
        m = MUTATION_SEGMENT.match(part)
        if not m:
            continue
        key, old_val, new_val = m.group(1), m.group(2).strip(), m.group(3).strip()
        if key in MUTATION_SPACE:
            out.append((key, old_val, new_val))
    return out


def weighted_choice(rng: random.Random, items: list[str], raw_scores: list[float]) -> str:
    min_raw = min(raw_scores)
    weights = [max(0.01, s - min_raw + 0.03) for s in raw_scores]
    return rng.choices(items, weights=weights, k=1)[0]


def key_raw_score(stat: ArmStats, run_id: int, plateau_mode: bool) -> float:
    mean_delta = stat.delta_sum / stat.trials if stat.trials else 0.0
    success_rate = stat.keeps / stat.trials if stat.trials else 0.0
    explore_bonus = 0.40 / math.sqrt(stat.trials + 1)

    recency_penalty = 0.0
    if stat.last_used_run > 0 and (run_id - stat.last_used_run) <= 2:
        recency_penalty = 0.12

    plateau_boost = 0.10 if plateau_mode and stat.trials <= 2 else 0.0
    return mean_delta * 35.0 + success_rate * 0.9 + explore_bonus + plateau_boost - recency_penalty


def value_raw_score(
    key: str,
    candidate: str,
    current: str,
    stat: ArmStats,
    run_id: int,
    plateau_mode: bool,
    tabu: TabuMemory,
    base_commit: str,
) -> float:
    mean_delta = stat.delta_sum / stat.trials if stat.trials else 0.0
    success_rate = stat.keeps / stat.trials if stat.trials else 0.0
    explore_bonus = 0.34 / math.sqrt(stat.trials + 1)

    values = MUTATION_SPACE[key]
    neighbor_bonus = 0.0
    if current in values:
        idx = values.index(current)
        neighbors: set[str] = set()
        if idx - 1 >= 0:
            neighbors.add(values[idx - 1])
        if idx + 1 < len(values):
            neighbors.add(values[idx + 1])
        if candidate in neighbors:
            neighbor_bonus = 0.15 if not plateau_mode else 0.05

    recency_penalty = 0.0
    if stat.last_used_run > 0 and (run_id - stat.last_used_run) <= 3:
        recency_penalty = 0.08

    tabu_penalty = 0.90 if tabu.contains(mutation_signature(base_commit, key, candidate)) else 0.0

    return mean_delta * 30.0 + success_rate * 0.8 + explore_bonus + neighbor_bonus - recency_penalty - tabu_penalty


def choose_key(
    rng: random.Random,
    key_stats: dict[str, ArmStats],
    run_id: int,
    plateau_mode: bool,
    exclude: set[str],
) -> str:
    keys = [k for k in MUTATION_SPACE.keys() if k not in exclude]
    raw_scores = [key_raw_score(key_stats[k], run_id, plateau_mode) for k in keys]
    return weighted_choice(rng, keys, raw_scores)


def choose_value(
    rng: random.Random,
    key: str,
    current_val: str,
    value_stats: dict[tuple[str, str], ArmStats],
    run_id: int,
    plateau_mode: bool,
    tabu: TabuMemory,
    base_commit: str,
) -> str:
    candidates = [v for v in MUTATION_SPACE[key] if v != current_val]
    if not candidates:
        raise RuntimeError(f"No candidate value for {key}")

    raw_scores = [
        value_raw_score(
            key=key,
            candidate=c,
            current=current_val,
            stat=value_stats[(key, c)],
            run_id=run_id,
            plateau_mode=plateau_mode,
            tabu=tabu,
            base_commit=base_commit,
        )
        for c in candidates
    ]

    return weighted_choice(rng, candidates, raw_scores)


def choose_mutation_plan(
    rng: random.Random,
    current_text: str,
    base_commit: str,
    key_stats: dict[str, ArmStats],
    value_stats: dict[tuple[str, str], ArmStats],
    tabu: TabuMemory,
    run_id: int,
    runs_since_improve: int,
    plateau_runs: int,
    two_mutation_prob: float,
) -> tuple[list[tuple[str, str, str]], str, bool]:
    plateau_mode = runs_since_improve >= plateau_runs
    mutation_count = 1
    if plateau_mode and rng.random() < two_mutation_prob:
        mutation_count = 2

    working_text = current_text
    used_keys: set[str] = set()
    mutations: list[tuple[str, str, str]] = []

    for _ in range(mutation_count):
        key = choose_key(rng, key_stats, run_id, plateau_mode, exclude=used_keys)
        current_val = parse_current_literal(working_text, key)
        target_val = choose_value(
            rng,
            key,
            current_val,
            value_stats,
            run_id,
            plateau_mode,
            tabu,
            base_commit,
        )
        working_text, old_val = replace_literal_in_text(working_text, key, target_val)
        mutations.append((key, old_val, target_val))
        used_keys.add(key)

    return mutations, working_text, plateau_mode


def record_feedback(
    mutations: list[tuple[str, str, str]],
    delta: float,
    kept: bool,
    run_id: int,
    key_stats: dict[str, ArmStats],
    value_stats: dict[tuple[str, str], ArmStats],
) -> None:
    for key, _old, new in mutations:
        ks = key_stats[key]
        ks.trials += 1
        ks.keeps += 1 if kept else 0
        ks.delta_sum += delta
        ks.last_used_run = run_id

        vs = value_stats[(key, new)]
        vs.trials += 1
        vs.keeps += 1 if kept else 0
        vs.delta_sum += delta
        vs.last_used_run = run_id


def bootstrap_stats_from_results(
    path: Path,
    key_stats: dict[str, ArmStats],
    value_stats: dict[tuple[str, str], ArmStats],
    tabu: TabuMemory,
) -> tuple[int, int]:
    if not path.exists():
        return 0, 0

    run_id = 0
    runs_since_improve = 0
    best = float("-inf")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run_id += 1
            status = (row.get("status") or "").strip().lower()
            desc = (row.get("description") or "").strip()
            commit = (row.get("commit") or "").strip()

            try:
                ndcg = float(row.get("val_ndcg_at_10", "0") or "0")
            except ValueError:
                ndcg = 0.0

            mutations = parse_mutations_from_description(desc)

            best_before = best
            delta = ndcg - best_before if best_before > -1e8 else 0.0
            kept = status == "keep"

            if mutations:
                record_feedback(mutations, delta, kept, run_id, key_stats, value_stats)
                for key, _old, new in mutations:
                    if commit:
                        tabu.add(mutation_signature(commit, key, new))

            if kept and ndcg > best + 1e-12:
                best = ndcg
                runs_since_improve = 0
            else:
                if best == float("-inf") and ndcg > best:
                    best = ndcg
                runs_since_improve += 1

    return run_id, runs_since_improve


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autonomous retrieval experiment loop")
    parser.add_argument("--repo", default=".", help="Repo root")
    parser.add_argument("--hours", type=float, default=7.0, help="Loop duration in hours")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional hard cap on runs (0 = unlimited)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--improve-eps", type=float, default=1e-6)
    parser.add_argument("--plateau-runs", type=int, default=8)
    parser.add_argument("--two-mutation-prob", type=float, default=0.35)
    parser.add_argument("--tabu-size", type=int, default=300)
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

    key_stats = {k: ArmStats() for k in MUTATION_SPACE.keys()}
    value_stats = {(k, v): ArmStats() for k, values in MUTATION_SPACE.items() for v in values}
    tabu = TabuMemory(args.tabu_size)

    global_run_id, runs_since_improve = bootstrap_stats_from_results(results_path, key_stats, value_stats, tabu)

    session_run_idx = 0
    keep_count = 0

    log_line(f"[loop] repo={repo}")
    log_line(f"[loop] starting best nDCG@10={best_ndcg:.6f}" if best_ndcg > -1e8 else "[loop] no prior best")
    log_line(f"[loop] deadline in {args.hours:.2f}h")
    log_line(
        f"[loop] policy=adaptive plateau_runs={args.plateau_runs} two_mutation_prob={args.two_mutation_prob:.2f} "
        f"tabu_size={args.tabu_size}"
    )

    while time.time() < deadline:
        if args.max_runs > 0 and session_run_idx >= args.max_runs:
            break

        session_run_idx += 1
        global_run_id += 1

        head_before = git_short_head(repo)
        current_text = train_path.read_text(encoding="utf-8")

        mutations, mutated_text, plateau_mode = choose_mutation_plan(
            rng=rng,
            current_text=current_text,
            base_commit=head_before,
            key_stats=key_stats,
            value_stats=value_stats,
            tabu=tabu,
            run_id=global_run_id,
            runs_since_improve=runs_since_improve,
            plateau_runs=args.plateau_runs,
            two_mutation_prob=args.two_mutation_prob,
        )

        train_path.write_text(mutated_text, encoding="utf-8")
        desc = " ; ".join(f"{k} {old} -> {new}" for k, old, new in mutations)
        mode_tag = "plateau" if plateau_mode else "search"

        log_line(f"[run {session_run_idx:03d}] start | base={head_before} | mode={mode_tag} | {desc}")
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
        best_before = best_ndcg

        if ndcg is None:
            train_path.write_text(current_text, encoding="utf-8")
            fail_hint = "timeout" if proc.returncode == 124 else f"exit={proc.returncode}"
            outcome = RunOutcome(
                status="crash",
                val_ndcg=0.0,
                peak_vram_mb=0.0,
                description=f"{desc} ({fail_hint})",
                commit=f"wip{global_run_id:04d}",
            )
            append_result(results_path, outcome)
            record_feedback(mutations, delta=-0.05, kept=False, run_id=global_run_id, key_stats=key_stats, value_stats=value_stats)
            for key, _old, new in mutations:
                tabu.add(mutation_signature(head_before, key, new))
            runs_since_improve += 1
            log_line(f"[run {session_run_idx:03d}] crash | {desc} | {fail_hint}")
            time.sleep(1.0)
            continue

        if peak_vram is None:
            peak_vram = 0.0

        delta = ndcg - best_before if best_before > -1e8 else 0.0

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
            record_feedback(mutations, delta=delta, kept=True, run_id=global_run_id, key_stats=key_stats, value_stats=value_stats)
            for key, _old, new in mutations:
                tabu.add(mutation_signature(head_before, key, new))
            runs_since_improve = 0
            log_line(
                f"[run {session_run_idx:03d}] keep    | ndcg={ndcg:.6f} | delta={delta:+.6f} "
                f"| vram={peak_vram/1024:.1f} GB | commit={commit}"
            )
        else:
            train_path.write_text(current_text, encoding="utf-8")
            outcome = RunOutcome(
                status="discard",
                val_ndcg=ndcg,
                peak_vram_mb=peak_vram,
                description=desc,
                commit=f"wip{global_run_id:04d}",
            )
            append_result(results_path, outcome)
            record_feedback(mutations, delta=delta, kept=False, run_id=global_run_id, key_stats=key_stats, value_stats=value_stats)
            for key, _old, new in mutations:
                tabu.add(mutation_signature(head_before, key, new))
            runs_since_improve += 1
            log_line(
                f"[run {session_run_idx:03d}] discard | ndcg={ndcg:.6f} (best={best_ndcg:.6f}) "
                f"| delta={delta:+.6f} | vram={peak_vram/1024:.1f} GB"
            )

    elapsed_h = (time.time() - start) / 3600.0
    log_line(f"[loop] finished runs={session_run_idx} keeps={keep_count} elapsed={elapsed_h:.2f}h best={best_ndcg:.6f}")


if __name__ == "__main__":
    main()
