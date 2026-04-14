"""
Training and evaluation functions for PURS model.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def _binary_auc_from_single_positive(scores: np.ndarray, positive_index: int) -> float:
    """Compute AUC for one positive item versus multiple negatives."""
    if len(scores) <= 1:
        return 0.5

    pos_score = scores[positive_index]
    neg_scores = np.delete(scores, positive_index)

    greater = np.sum(pos_score > neg_scores)
    equal = np.sum(pos_score == neg_scores)

    return float((greater + 0.5 * equal) / len(neg_scores))


def train_purs(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    verbose: bool = True,
):
    """
    Train PURS model for one epoch using pointwise binary cross-entropy.

    Args:
        model: PURS model
        train_loader: DataLoader with training samples (user, [history], item, label)
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        verbose: If True, show progress bar

    Returns:
        Average loss for the epoch
    """
    model.train()
    criterion = torch.nn.BCELoss()
    total_loss = 0.0
    num_batches = 0

    iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if verbose else train_loader

    for batch in iterator:
        # Unpack batch: (users, [histories], items, labels)
        # With history: (users_tensor, histories_tensor, items_tensor, labels_tensor)
        # Without history: (users_tensor, items_tensor, labels_tensor)

        if len(batch) == 4:
            users, histories, items, labels = batch
            histories = histories.to(device)
        else:
            users, items, labels = batch
            # Create dummy history if not provided
            batch_size = users.shape[0]
            histories = torch.zeros(batch_size, model.history_length, dtype=torch.long).to(device)

        users = users.view(-1).to(device)
        items = items.view(-1).to(device)
        labels = labels.view(-1).float().to(device)

        # Pointwise prediction probability and BCE loss.
        scores = model.forward(users, items, histories, compute_unexpectedness=False)
        loss = criterion(scores, labels)

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
    k_values: list = [10],
    verbose: bool = True,
):
    """
    Evaluate PURS model using candidate ranking with binary pointwise scores.

    Args:
        model: PURS model
        eval_loader: DataLoader with evaluation samples (user, [history], candidates, gt_position)
        device: Device to evaluate on
        k_values: List of K values for metrics@K
        verbose: If True, show progress bar

    Returns:
        Dictionary with metrics: hr@K, precision@K, ndcg@K, auc,
        unexpectedness@K, coverage@K, serendipity@K
    """
    model.eval()

    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"hr@{k}": [] for k in k_values})
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    metrics.update({f"unexpectedness@{k}": [] for k in k_values})
    metrics.update({f"serendipity@{k}": [] for k in k_values})
    metrics["auc"] = []

    coverage_items = {k: set() for k in k_values}

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

            users = users.view(-1).to(device)
            candidates = candidates.to(device)
            gt_positions = gt_positions.view(-1)

            batch_size = users.shape[0]
            top_k_max = max(k_values)

            # Rank candidates for each user
            for i in range(batch_size):
                user_id = users[i].item()
                candidate_items = candidates[i]
                gt_pos = gt_positions[i].item()

                # Get user history
                user_history = histories[i : i + 1]  # (1, history_length)

                # Score candidate set for this user.
                num_candidates = candidate_items.shape[0]
                user_ids_tensor = torch.full(
                    (num_candidates,), user_id, dtype=torch.long, device=device
                )
                history_batch = user_history.repeat(num_candidates, 1)

                scores = (
                    model.forward(
                        user_ids_tensor,
                        candidate_items,
                        history_batch,
                        compute_unexpectedness=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                candidate_items_np = candidate_items.detach().cpu().numpy()

                # Rank candidates by score (descending)
                ranking = np.argsort(-scores)[:top_k_max]

                # AUC for one-positive candidate set.
                metrics["auc"].append(_binary_auc_from_single_positive(scores, gt_pos))

                history_items = set(user_history[0][user_history[0] > 0].cpu().numpy().tolist())
                relevant_items = {int(candidate_items_np[gt_pos])}

                # Compute metrics for each K
                for k in k_values:
                    ranking_k = ranking[:k]
                    hit = 1 if gt_pos in ranking_k else 0

                    top_items_k = [int(x) for x in candidate_items_np[ranking_k].tolist()]

                    # HR@K, Precision@K, Recall@K
                    metrics[f"hr@{k}"].append(float(hit))
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

                    # Unexpectedness@K: ratio of top-K items not in user history.
                    unexpected_items = [item for item in top_items_k if item not in history_items]
                    unexpectedness = len(unexpected_items) / k if k > 0 else 0.0
                    metrics[f"unexpectedness@{k}"].append(unexpectedness)

                    # Serendipity@K: recommended items that are both relevant and unexpected.
                    serendip_items = [
                        item for item in top_items_k if item in relevant_items and item not in history_items
                    ]
                    metrics[f"serendipity@{k}"].append(len(serendip_items) / k if k > 0 else 0.0)

                    # Coverage@K: unique recommended items across users.
                    coverage_items[k].update(top_items_k)

    # Average metrics
    avg_metrics = {key: np.mean(values) if values else 0.0 for key, values in metrics.items()}

    # Coverage is dataset-level, not user-level average.
    num_items = max(int(getattr(model, "num_items", 0)), 1)
    for k in k_values:
        avg_metrics[f"coverage@{k}"] = len(coverage_items[k]) / num_items

    return avg_metrics
