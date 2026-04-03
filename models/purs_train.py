"""
Training and evaluation functions for PURS model.
"""

import torch
import numpy as np
from typing import Dict, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_purs(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    verbose: bool = True,
):
    """
    Train PURS model for one epoch using triplet loss.

    Args:
        model: PURS model
        train_loader: DataLoader with training triplets + histories
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        verbose: If True, show progress bar

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if verbose else train_loader

    for batch in iterator:
        # Unpack batch: (users, [histories], pos_items, neg_items)
        # With history: (users_tensor, histories_tensor, pos_items_tensor, neg_items_tensor)
        # Without history: (users_tensor, pos_items_tensor, neg_items_tensor)

        if len(batch) == 4:
            users, histories, items_pos, items_neg = batch
            histories = histories.to(device)
        else:
            users, items_pos, items_neg = batch
            # Create dummy history if not provided
            batch_size = users.shape[0]
            histories = torch.zeros(batch_size, model.history_length, dtype=torch.long).to(device)

        users = users.squeeze().to(device)
        items_pos = items_pos.squeeze().to(device)
        items_neg = items_neg.squeeze().to(device)

        # Forward pass - get scores for positive items
        pos_scores = model.forward(users, items_pos, histories, compute_unexpectedness=True)

        # Get scores for negative items
        neg_scores = model.forward(users, items_neg, histories, compute_unexpectedness=True)

        # Triplet loss: positive score should be higher than negative
        # Loss = -log(sigmoid(pos_score - neg_score))
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        if verbose and isinstance(iterator, tqdm):
            iterator.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate_purs(
    model,
    eval_loader: DataLoader,
    device: str,
    k_values: list = [5, 10, 20],
    verbose: bool = True,
):
    """
    Evaluate PURS model using candidate ranking.

    Args:
        model: PURS model
        eval_loader: DataLoader with evaluation samples (user, [history], candidates, gt_position)
        device: Device to evaluate on
        k_values: List of K values for metrics@K
        verbose: If True, show progress bar

    Returns:
        Dictionary with metrics: precision@K, recall@K, ndcg@K, unexpectedness@K
    """
    model.eval()

    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics.update({f"unexpectedness@{k}": [] for k in k_values})  # NEW: Unexpectedness metric

    iterator = tqdm(eval_loader, desc="Evaluating") if verbose else eval_loader

    with torch.no_grad():
        for batch in iterator:
            # Unpack batch
            if len(batch) == 4:
                users, histories, candidates, gt_positions = batch
                histories = histories.to(device)
            else:
                users, candidates, gt_positions = batch
                batch_size = users.shape[0]
                histories = torch.zeros(batch_size, model.history_length, dtype=torch.long).to(device)

            users = users.squeeze().to(device)
            candidates = candidates.to(device)
            gt_positions = gt_positions.squeeze()

            batch_size = users.shape[0]
            top_k_max = max(k_values)

            # Rank candidates for each user
            for i in range(batch_size):
                user_id = users[i].item()
                candidate_items = candidates[i]
                gt_pos = gt_positions[i].item()

                # Get user history
                user_history = histories[i : i + 1]  # (1, history_length)

                # Score each candidate
                scores = []
                for candidate_idx, item_id in enumerate(candidate_items):
                    user_id_tensor = torch.LongTensor([user_id]).to(device)
                    item_id_tensor = torch.LongTensor([item_id]).to(device)
                    score = model.forward(
                        user_id_tensor, item_id_tensor, user_history, compute_unexpectedness=True
                    )
                    scores.append(score.item())

                scores = np.array(scores)

                # Rank candidates by score (descending)
                ranking = np.argsort(-scores)[:top_k_max]

                # Compute metrics for each K
                for k in k_values:
                    ranking_k = ranking[:k]
                    hit = 1 if gt_pos in ranking_k else 0

                    # Precision@K and Recall@K
                    metrics[f"precision@{k}"].append(hit / k)
                    metrics[f"recall@{k}"].append(hit)  # Single positive item

                    # NDCG@K
                    if hit:
                        pos_in_ranking = np.where(ranking_k == gt_pos)[0][0]
                        dcg = 1.0 / np.log2(pos_in_ranking + 2)
                        idcg = 1.0 / np.log2(2)
                        ndcg = dcg / idcg
                    else:
                        ndcg = 0.0

                    metrics[f"ndcg@{k}"].append(ndcg)

                    # Unexpectedness@K: fraction of top-K items not in user history
                    if len(user_history) > 0:
                        history_items = set(user_history[0][user_history[0] > 0].cpu().numpy())
                        ranked_items_k = set(candidate_items[ranking_k].cpu().numpy())
                        unexpected_count = len(ranked_items_k - history_items)
                        unexpectedness = unexpected_count / k if k > 0 else 0.0
                    else:
                        unexpectedness = 1.0  # All items are unexpected if empty history

                    metrics[f"unexpectedness@{k}"].append(unexpectedness)

    # Average metrics
    avg_metrics = {key: np.mean(values) if values else 0.0 for key, values in metrics.items()}

    return avg_metrics
