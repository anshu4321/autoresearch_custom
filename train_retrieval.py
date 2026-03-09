"""
Autoresearch retrieval training script. Single-GPU, single-file.
Usage: uv run train_retrieval.py
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from prepare_retrieval import TIME_BUDGET, evaluate_retrieval, load_runtime_data

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_QUERY_LEN = 48
MAX_DOC_LEN = 256
PROJECTION_DIM = 256

TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
GRAD_CLIP_NORM = 1.0

UNFREEZE_LAST_N = 2  # 0 = freeze full backbone, -1 = train full backbone
SEED = 42


@dataclass
class TrainConfig:
    model_name: str
    max_query_len: int
    max_doc_len: int
    projection_dim: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    temperature: float
    grad_clip_norm: float
    unfreeze_last_n: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def _get_transformer_layers(backbone: nn.Module):
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    if hasattr(backbone, "transformer") and hasattr(backbone.transformer, "layer"):
        return backbone.transformer.layer
    if hasattr(backbone, "model") and hasattr(backbone.model, "layers"):
        return backbone.model.layers
    return None


def _configure_trainable_layers(backbone: nn.Module, unfreeze_last_n: int) -> None:
    for param in backbone.parameters():
        param.requires_grad = False

    if unfreeze_last_n < 0:
        for param in backbone.parameters():
            param.requires_grad = True
        return

    if unfreeze_last_n == 0:
        return

    layers = _get_transformer_layers(backbone)
    if layers is None:
        for param in backbone.parameters():
            param.requires_grad = True
        return

    for layer in layers[-unfreeze_last_n:]:
        for param in layer.parameters():
            param.requires_grad = True


class DualEncoder(nn.Module):
    def __init__(self, model_name: str, projection_dim: int, temperature: float, unfreeze_last_n: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.backbone.config.hidden_size
        self.proj = nn.Linear(self.hidden_dim, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))
        _configure_trainable_layers(self.backbone, unfreeze_last_n)

    def encode(self, batch_tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.backbone(**batch_tokens)
        emb = _mean_pool(out.last_hidden_state, batch_tokens["attention_mask"])
        emb = self.proj(emb)
        emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, q_tokens: dict[str, torch.Tensor], d_tokens: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.encode(q_tokens)
        d = self.encode(d_tokens)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * (q @ d.T)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_q2d = F.cross_entropy(logits, labels)
        loss_d2q = F.cross_entropy(logits.T, labels)
        loss = 0.5 * (loss_q2d + loss_d2q)
        return loss, logits


class PairSampler:
    def __init__(self, queries: list[str], docs: list[str], seed: int):
        if len(queries) != len(docs):
            raise ValueError("queries and docs must have the same length")
        if not queries:
            raise ValueError("empty train set")
        self.queries = queries
        self.docs = docs
        self.rng = random.Random(seed)

    def sample(self, batch_size: int) -> tuple[list[str], list[str]]:
        idxs = [self.rng.randrange(len(self.queries)) for _ in range(batch_size)]
        q = [self.queries[i] for i in idxs]
        d = [self.docs[i] for i in idxs]
        return q, d


def _tokenize(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device, non_blocking=True) for k, v in tokens.items()}


@torch.no_grad()
def _encode_texts(
    model: DualEncoder,
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> torch.Tensor:
    embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_tokens = _tokenize(tokenizer, batch_texts, max_length, device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            batch_emb = model.encode(batch_tokens)
        embs.append(batch_emb.float().cpu())
    return torch.cat(embs, dim=0)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")

    config = TrainConfig(
        model_name=MODEL_NAME,
        max_query_len=MAX_QUERY_LEN,
        max_doc_len=MAX_DOC_LEN,
        projection_dim=PROJECTION_DIM,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        temperature=TEMPERATURE,
        grad_clip_norm=GRAD_CLIP_NORM,
        unfreeze_last_n=UNFREEZE_LAST_N,
    )

    t_start = time.time()
    _set_seed(SEED)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    data = load_runtime_data()
    print(f"Train pairs: {len(data.train_queries):,}")
    print(f"Val queries: {len(data.val_queries):,}")
    print(f"Corpus docs: {len(data.doc_ids):,}")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Config: {asdict(config)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = DualEncoder(
        model_name=config.model_name,
        projection_dim=config.projection_dim,
        temperature=config.temperature,
        unfreeze_last_n=config.unfreeze_last_n,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters left. Increase UNFREEZE_LAST_N or adjust model setup.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    sampler = PairSampler(data.train_queries, data.train_docs, seed=SEED)

    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0

    model.train()
    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        q_texts, d_texts = sampler.sample(config.train_batch_size)
        q_tokens = _tokenize(tokenizer, q_texts, config.max_query_len, device)
        d_tokens = _tokenize(tokenizer, d_texts, config.max_doc_len, device)

        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            loss, _ = model(q_tokens, d_tokens)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip_norm)
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0

        if step > 2:
            total_training_time += dt

        loss_f = loss.item()
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_f
        corrected = smooth_loss / (1 - 0.9 ** (step + 1))
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:05d} ({100*progress:.1f}%) | "
            f"loss: {corrected:.4f} | dt: {dt*1000:.0f}ms | "
            f"remaining: {remaining:.0f}s",
            end="",
            flush=True,
        )

        step += 1
        if step > 2 and total_training_time >= TIME_BUDGET:
            break

    print()

    model.eval()
    with torch.no_grad():
        doc_embeddings = _encode_texts(
            model,
            tokenizer,
            data.doc_texts,
            max_length=config.max_doc_len,
            batch_size=config.eval_batch_size,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        query_embeddings = _encode_texts(
            model,
            tokenizer,
            data.val_queries,
            max_length=config.max_query_len,
            batch_size=config.eval_batch_size,
            device=device,
            autocast_dtype=autocast_dtype,
        )

    metrics = evaluate_retrieval(
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        query_ids=data.val_query_ids,
        doc_ids=data.doc_ids,
        qrels=data.val_qrels,
        k=10,
    )

    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    num_trainable_params = sum(p.numel() for p in trainable_params)

    print("---")
    print(f"val_ndcg_at_10:   {metrics['ndcg@10']:.6f}")
    print(f"val_recall_at_10: {metrics['recall@10']:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"trainable_params: {num_trainable_params / 1e6:.2f}M")
    print(f"model_name:       {MODEL_NAME}")


if __name__ == "__main__":
    main()
