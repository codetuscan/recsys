"""
Training and evaluation functions for SASRec.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_sasrec(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    verbose: bool = True,
):
    """Train SASRec model for one epoch with pairwise BCE objective."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if verbose else train_loader

    for batch in iterator:
        if len(batch) != 4:
            raise ValueError("SASRec expects batches: (users, histories, pos_items, neg_items)")

        _, histories, items_pos, items_neg = batch
        histories = histories.to(device)
        items_pos = items_pos.view(-1).to(device)
        items_neg = items_neg.view(-1).to(device)

        pos_scores = model(histories, items_pos)
        neg_scores = model(histories, items_neg)

        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss

        if model.reg_lambda > 0:
            reg_loss = model.reg_lambda * model.item_embedding.weight.norm(2).pow(2) / (model.num_items + 1)
            loss = loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, num_batches)


def evaluate_sasrec(
    model,
    eval_loader: DataLoader,
    device: str,
    k_values: list[int] = [5, 10, 20],
    verbose: bool = True,
):
    """
    Evaluate SASRec model using candidate ranking.

    Returns precision@K, recall@K, ndcg@K, unexpectedness@K.
    """
    model.eval()

    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics.update({f"unexpectedness@{k}": [] for k in k_values})

    iterator = tqdm(eval_loader, desc="Evaluating") if verbose else eval_loader

    top_k_max = max(k_values)

    with torch.no_grad():
        for batch in iterator:
            if len(batch) == 4:
                _, histories, candidates, gt_positions = batch
                histories = histories.to(device)
            else:
                # Fallback for non-history batch structure.
                _, candidates, gt_positions = batch
                batch_size = candidates.shape[0]
                histories = torch.full(
                    (batch_size, model.max_seq_length),
                    fill_value=model.pad_token,
                    dtype=torch.long,
                    device=device,
                )

            candidates = candidates.to(device)
            gt_positions = gt_positions.view(-1).cpu().numpy()

            scores = model.score_items(histories, candidates)  # (batch_size, num_candidates)
            top_idx = torch.topk(scores, k=top_k_max, dim=1).indices.cpu().numpy()

            candidates_cpu = candidates.cpu().numpy()
            histories_cpu = histories.cpu().numpy()

            batch_size = candidates_cpu.shape[0]
            for i in range(batch_size):
                gt_pos = int(gt_positions[i])
                ranked_candidate_positions = top_idx[i]

                for k in k_values:
                    ranking_k = ranked_candidate_positions[:k]
                    hit = int(gt_pos in ranking_k)

                    metrics[f"precision@{k}"].append(hit / k)
                    metrics[f"recall@{k}"].append(float(hit))

                    if hit:
                        pos_in_ranking = int(np.where(ranking_k == gt_pos)[0][0])
                        ndcg = 1.0 / np.log2(pos_in_ranking + 2)
                    else:
                        ndcg = 0.0
                    metrics[f"ndcg@{k}"].append(ndcg)

                    history_items = set(histories_cpu[i][histories_cpu[i] != model.pad_token])
                    ranked_items_k = set(candidates_cpu[i][ranking_k])
                    unexpected_count = len(ranked_items_k - history_items)
                    unexpectedness = unexpected_count / k if k > 0 else 0.0
                    metrics[f"unexpectedness@{k}"].append(unexpectedness)

    return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}
