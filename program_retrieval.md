# autoresearch-retrieval

This is an autoresearch program for dense retrieval (dual-encoder) experiments.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: suggest a date-style tag (e.g. `mar9-retrieval`).
2. **Create branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `README.md`
   - `prepare_retrieval.py` (fixed prep + eval harness, do not modify)
   - `train_retrieval.py` (the editable file)
4. **Verify data exists**: make sure `~/.cache/autoresearch_retrieval/prepared_scifact.pt` exists.
   - If missing, run: `uv run prepare_retrieval.py`
5. **Initialize results file**: create `results_retrieval.tsv` with header only.
6. **Confirm and go**.

## Experimentation

Each run is a single-GPU experiment with a **fixed 5-minute training budget**. Launch command:

```bash
uv run train_retrieval.py
```

**What you CAN do:**
- Modify `train_retrieval.py` only.

**What you CANNOT do:**
- Do not modify `prepare_retrieval.py`.
- Do not change the evaluation function `evaluate_retrieval`.
- Do not add dependencies during experiment loop.

## Objective

Maximize `val_ndcg_at_10` (higher is better). Use `val_recall_at_10` as a secondary metric.

## Output format

The training script prints:

```text
---
val_ndcg_at_10:   0.123456
val_recall_at_10: 0.234567
training_seconds: 300.0
total_seconds:    340.0
peak_vram_mb:     12345.6
num_steps:        123
trainable_params: 25.00M
model_name:       sentence-transformers/all-MiniLM-L6-v2
```

## Logging results

Use `results_retrieval.tsv` with tab-separated columns:

```text
commit	val_ndcg_at_10	memory_gb	status	description
```

- `status` is one of: `keep`, `discard`, `crash`
- if crash, use `0.000000` for metric and `0.0` for memory

## Loop

1. Inspect current git state.
2. Edit `train_retrieval.py` with one clear hypothesis.
3. Commit.
4. Run: `uv run train_retrieval.py > run.log 2>&1`
5. Extract metrics: `grep "^val_ndcg_at_10:\|^peak_vram_mb:" run.log`
6. If no metric line, inspect crash: `tail -n 80 run.log`
7. Append result row to `results_retrieval.tsv` (untracked).
8. If `val_ndcg_at_10` improves, keep commit.
9. If not, reset to prior best commit.

Timeout rule: if runtime exceeds 10 minutes, treat as failed and discard.
