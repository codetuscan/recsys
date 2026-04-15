"""
Training and evaluation functions for PURS model.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def _public_auc_from_records(records: list[list[float]]) -> float:
    """AUC calculation aligned with the public PURS implementation."""
    if not records:
        return 0.0

    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0

    arr = sorted(records, key=lambda d: d[2])
    for record in arr:
        fp2 += record[0]  # noclick
        tp2 += record[1]  # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5
    if tp2 * fp2 > 0.0:
        return float(1.0 - auc / (2.0 * tp2 * fp2))
    return 0.0


def _public_hit_rate_from_records(records: list[list[float]]) -> float:
    """Hit-rate calculation aligned with the public PURS implementation."""
    if not records:
        return 0.0

    hit_values = []
    user_ids = sorted({int(x[2]) for x in records})

    for user in user_ids:
        arr_user = [x for x in records if int(x[2]) == user and int(x[1]) == 1]
        # Public code divides by len(arr_user); guard empty-user case to avoid runtime errors.
        if not arr_user:
            hit_values.append(0.0)
            continue
        hit_values.append(float(sum(x[0] for x in arr_user) / len(arr_user)))

    return float(np.mean(hit_values)) if hit_values else 0.0


def _unexpectedness_scores(model, item_ids: torch.Tensor, histories: torch.Tensor) -> list[float]:
    """Compute per-sample unexpectedness values for metric logging."""
    scores = []
    for i in range(item_ids.shape[0]):
        user_history = histories[i]
        candidate_item = int(item_ids[i].item())

        centroids, cluster_sizes = model.compute_user_clusters(user_history)
        candidate_embedding = (
            model.item_embedding(item_ids[i : i + 1]).detach().cpu().numpy()[0]
        )
        unexp = model.compute_unexpectedness(candidate_embedding, centroids, cluster_sizes)
        scores.append(float(unexp))
    return scores


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
    Evaluate PURS model using public PURS-style metrics.

    Args:
        model: PURS model
        eval_loader: DataLoader with evaluation samples (user, [history], item, label)
        device: Device to evaluate on
        k_values: Unused placeholder kept for function signature compatibility
        verbose: If True, show progress bar

    Returns:
        Dictionary with public-style metric keys.
    """
    model.eval()

    auc_records: list[list[float]] = []
    hit_records: list[list[float]] = []
    rec_items: list[int] = []
    unexpectedness_values: list[float] = []

    iterator = tqdm(eval_loader, desc="Evaluating") if verbose else eval_loader

    with torch.no_grad():
        for batch in iterator:
            # Unpack batch
            if len(batch) == 4:
                users, histories, items, labels = batch
                histories = histories.to(device)
            else:
                users, items, labels = batch
                batch_size = users.shape[0]
                histories = torch.zeros(batch_size, model.history_length, dtype=torch.long).to(device)

            users = users.view(-1).to(device)
            items = items.view(-1).to(device)
            labels = labels.view(-1).float().to(device)

            scores = (
                model.forward(users, items, histories, compute_unexpectedness=False)
                .detach()
                .cpu()
                .numpy()
            )
            labels_np = labels.detach().cpu().numpy()
            users_np = users.detach().cpu().numpy()
            items_np = items.detach().cpu().numpy()
            preds_np = (scores > 0.5).astype(np.int32)

            for label_val, score_val in zip(labels_np, scores):
                if label_val > 0:
                    auc_records.append([0.0, 1.0, float(score_val)])
                else:
                    auc_records.append([1.0, 0.0, float(score_val)])

            for label_val, pred_val, user_id in zip(labels_np, preds_np, users_np):
                hit_records.append([float(label_val), int(pred_val), int(user_id)])

            for pred_val, item_id in zip(preds_np, items_np):
                if pred_val == 1:
                    rec_items.append(int(item_id))

            unexpectedness_values.extend(_unexpectedness_scores(model, items, histories))

    hit_rate = _public_hit_rate_from_records(hit_records)
    auc_value = _public_auc_from_records(auc_records)
    coverage_value = len(set(rec_items)) / max(int(getattr(model, "num_items", 0)), 1)
    unexpectedness_value = float(np.mean(unexpectedness_values)) if unexpectedness_values else 0.0

    _ = k_values

    return {
        "auc": auc_value,
        "hit_rate": hit_rate,
        # In the public implementation, hit_rate is effectively precision over predicted positives.
        "precision": hit_rate,
        "coverage": coverage_value,
        "unexpectedness": unexpectedness_value,
    }
