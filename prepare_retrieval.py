"""
One-time data preparation for retrieval autoresearch experiments.
Downloads a BEIR dataset (SciFact by default), builds a deterministic
train/validation query split, and exposes fixed evaluation utilities.

Usage:
    uv run prepare_retrieval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List

import requests
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify during experiments)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # 5 minutes

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_retrieval")
DATA_DIR = os.path.join(CACHE_DIR, "data")
PREPARED_PATH = os.path.join(CACHE_DIR, "prepared_scifact.pt")

DATASET_NAME = "scifact"
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

SPLIT_SEED = 1337
VAL_QUERY_FRACTION = 0.20
VAL_QUERY_MIN = 64
VAL_QUERY_MAX = 256
MAX_TRAIN_PAIRS = 20_000


@dataclass
class RetrievalRuntimeData:
    train_queries: List[str]
    train_docs: List[str]
    val_query_ids: List[str]
    val_queries: List[str]
    doc_ids: List[str]
    doc_texts: List[str]
    val_qrels: Dict[str, Dict[str, int]]


def _safe_join_text(title: str, text: str) -> str:
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if text and text.strip():
        parts.append(text.strip())
    return "\n\n".join(parts)


def _download_file(url: str, dst_path: str, max_attempts: int = 5) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                tmp_path = dst_path + ".tmp"
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, dst_path)
            return
        except (requests.RequestException, OSError) as exc:
            print(f"Download attempt {attempt}/{max_attempts} failed: {exc}")
            for path in (dst_path + ".tmp", dst_path):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to download dataset from {url}")


def _ensure_dataset_present(force_redownload: bool = False) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    dataset_dir = os.path.join(DATA_DIR, DATASET_NAME)
    corpus_path = os.path.join(dataset_dir, "corpus.jsonl")
    queries_path = os.path.join(dataset_dir, "queries.jsonl")
    qrels_path = os.path.join(dataset_dir, "qrels", "test.tsv")

    if (
        not force_redownload
        and os.path.exists(corpus_path)
        and os.path.exists(queries_path)
        and os.path.exists(qrels_path)
    ):
        return dataset_dir

    zip_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.zip")
    if force_redownload and os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"Downloading {DATASET_NAME} from BEIR...")
    _download_file(DATASET_URL, zip_path)

    if os.path.exists(dataset_dir):
        for root, dirs, files in os.walk(dataset_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(dataset_dir)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    if not (os.path.exists(corpus_path) and os.path.exists(queries_path) and os.path.exists(qrels_path)):
        raise RuntimeError("Dataset extraction incomplete; expected BEIR files were not found.")

    return dataset_dir


def _load_corpus(path: str) -> Dict[str, str]:
    corpus: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["_id"])
            title = row.get("title", "")
            text = row.get("text", "")
            corpus[doc_id] = _safe_join_text(title, text)
    return corpus


def _load_queries(path: str) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    return queries


def _load_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = int(row["score"])
            if score <= 0:
                continue
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = score
    return qrels


def _make_query_split(valid_query_ids: List[str], seed: int) -> tuple[List[str], List[str]]:
    ids = list(valid_query_ids)
    random.Random(seed).shuffle(ids)
    n_total = len(ids)

    if n_total < 2:
        raise RuntimeError("Need at least 2 labeled queries to create train/validation splits.")

    n_val = int(round(n_total * VAL_QUERY_FRACTION))
    n_val = max(VAL_QUERY_MIN, n_val)
    n_val = min(VAL_QUERY_MAX, n_val)
    n_val = min(n_val, n_total - 1)

    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def _build_train_pairs(
    train_query_ids: List[str],
    queries: Dict[str, str],
    corpus: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    max_pairs: int,
    seed: int,
) -> tuple[List[str], List[str]]:
    pairs: List[tuple[str, str]] = []

    for qid in train_query_ids:
        query_text = queries[qid]
        positives = sorted(qrels[qid].items(), key=lambda x: x[1], reverse=True)
        for doc_id, _ in positives:
            if doc_id in corpus:
                pairs.append((query_text, corpus[doc_id]))

    if not pairs:
        raise RuntimeError("No training pairs could be constructed from qrels and corpus.")

    random.Random(seed).shuffle(pairs)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    train_queries = [p[0] for p in pairs]
    train_docs = [p[1] for p in pairs]
    return train_queries, train_docs


def prepare_runtime_data(force_redownload: bool = False) -> RetrievalRuntimeData:
    dataset_dir = _ensure_dataset_present(force_redownload=force_redownload)
    corpus = _load_corpus(os.path.join(dataset_dir, "corpus.jsonl"))
    queries = _load_queries(os.path.join(dataset_dir, "queries.jsonl"))
    qrels = _load_qrels(os.path.join(dataset_dir, "qrels", "test.tsv"))

    valid_query_ids = []
    for qid, rels in qrels.items():
        if qid not in queries:
            continue
        filtered = {doc_id: rel for doc_id, rel in rels.items() if doc_id in corpus and rel > 0}
        if filtered:
            qrels[qid] = filtered
            valid_query_ids.append(qid)

    valid_query_ids = sorted(valid_query_ids)
    train_qids, val_qids = _make_query_split(valid_query_ids, SPLIT_SEED)

    train_queries, train_docs = _build_train_pairs(
        train_qids, queries, corpus, qrels, MAX_TRAIN_PAIRS, SPLIT_SEED
    )

    doc_ids = sorted(corpus.keys())
    doc_texts = [corpus[doc_id] for doc_id in doc_ids]
    val_query_ids = sorted(val_qids)
    val_queries = [queries[qid] for qid in val_query_ids]
    val_qrels = {qid: qrels[qid] for qid in val_query_ids}

    runtime_data = RetrievalRuntimeData(
        train_queries=train_queries,
        train_docs=train_docs,
        val_query_ids=val_query_ids,
        val_queries=val_queries,
        doc_ids=doc_ids,
        doc_texts=doc_texts,
        val_qrels=val_qrels,
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(runtime_data.__dict__, PREPARED_PATH)
    return runtime_data


def load_runtime_data() -> RetrievalRuntimeData:
    if not os.path.exists(PREPARED_PATH):
        raise FileNotFoundError(
            f"Missing prepared data at {PREPARED_PATH}. Run `uv run prepare_retrieval.py` first."
        )

    payload = torch.load(PREPARED_PATH, map_location="cpu")
    return RetrievalRuntimeData(**payload)


def _dcg_at_k(relevances: List[int], k: int) -> float:
    score = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        gain = (2**rel - 1) / math.log2(rank + 1)
        score += gain
    return score


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    doc_embeddings: torch.Tensor,
    query_ids: List[str],
    doc_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> Dict[str, float]:
    """Fixed retrieval evaluation harness. Higher is better."""

    if query_embeddings.ndim != 2 or doc_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-2 tensors")

    if query_embeddings.size(0) != len(query_ids):
        raise ValueError("query_embeddings rows must match query_ids length")

    if doc_embeddings.size(0) != len(doc_ids):
        raise ValueError("doc_embeddings rows must match doc_ids length")

    query_embeddings = F.normalize(query_embeddings.float(), dim=-1)
    doc_embeddings = F.normalize(doc_embeddings.float(), dim=-1)

    sim = query_embeddings @ doc_embeddings.T
    topk = min(k, sim.size(1))

    ndcgs: List[float] = []
    recalls: List[float] = []

    for i, qid in enumerate(query_ids):
        rel_docs = qrels.get(qid, {})
        if not rel_docs:
            continue

        ranked_idx = torch.topk(sim[i], k=topk, largest=True).indices.tolist()
        ranked_doc_ids = [doc_ids[j] for j in ranked_idx]

        ranked_rels = [rel_docs.get(doc_id, 0) for doc_id in ranked_doc_ids]
        dcg = _dcg_at_k(ranked_rels, topk)

        ideal_rels = sorted(rel_docs.values(), reverse=True)
        idcg = _dcg_at_k(ideal_rels, topk)

        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        hits = sum(1 for rel in ranked_rels if rel > 0)
        recall = hits / max(1, len(rel_docs))

        ndcgs.append(ndcg)
        recalls.append(recall)

    return {
        "ndcg@10": float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0,
        "recall@10": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "num_eval_queries": float(len(ndcgs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BEIR retrieval data for autoresearch")
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload and re-extract dataset even if local files exist",
    )
    args = parser.parse_args()

    t0 = time.time()
    data = prepare_runtime_data(force_redownload=args.force_redownload)
    t1 = time.time()

    print("Prepared retrieval runtime data")
    print(f"  cache_dir:       {CACHE_DIR}")
    print(f"  prepared_path:   {PREPARED_PATH}")
    print(f"  train_pairs:     {len(data.train_queries):,}")
    print(f"  val_queries:     {len(data.val_queries):,}")
    print(f"  corpus_docs:     {len(data.doc_ids):,}")
    print(f"  took_seconds:    {t1 - t0:.1f}")


if __name__ == "__main__":
    main()
