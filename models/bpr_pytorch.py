"""
GPU-accelerated BPR (Bayesian Personalized Ranking) with PyTorch.

This implementation provides significant speedup over the NumPy version through:
- Mini-batch training
- GPU acceleration
- Efficient negative sampling
- Vectorized operations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Set, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm


class BPR_MF_PyTorch(nn.Module):
    """
    BPR Matrix Factorization model with PyTorch.

    Optimizes pairwise ranking: score(user, positive) > score(user, negative)
    Loss: -log(sigmoid(score_pos - score_neg)) + L2 regularization
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        reg_lambda: float = 0.01,
    ):
        """
        Initialize BPR-MF model.

        Args:
            num_users: Number of users
            num_items: Number of items
            embedding_dim: Dimensionality of latent embeddings
            reg_lambda: L2 regularization coefficient
        """
        super(BPR_MF_PyTorch, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.reg_lambda = reg_lambda

        # User and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings with Xavier initialization
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(
        self, users: torch.LongTensor, items_pos: torch.LongTensor, items_neg: torch.LongTensor
    ):
        """
        Forward pass: compute scores for positive and negative items.

        Args:
            users: User IDs (batch_size,)
            items_pos: Positive item IDs (batch_size,)
            items_neg: Negative item IDs (batch_size,)

        Returns:
            Tuple of (positive_scores, negative_scores)
        """
        # Get embeddings
        user_emb = self.user_embeddings(users)  # (batch_size, embedding_dim)
        pos_emb = self.item_embeddings(items_pos)  # (batch_size, embedding_dim)
        neg_emb = self.item_embeddings(items_neg)  # (batch_size, embedding_dim)

        # Compute scores via dot product
        pos_scores = (user_emb * pos_emb).sum(dim=1)  # (batch_size,)
        neg_scores = (user_emb * neg_emb).sum(dim=1)  # (batch_size,)

        return pos_scores, neg_scores

    def bpr_loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        """
        Compute BPR loss with L2 regularization.

        Args:
            pos_scores: Scores for positive items
            neg_scores: Scores for negative items

        Returns:
            Loss value
        """
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        # Add L2 regularization
        if self.reg_lambda > 0:
            reg_loss = self.reg_lambda * (
                self.user_embeddings.weight.norm(2).pow(2)
                + self.item_embeddings.weight.norm(2).pow(2)
            ) / (self.num_users + self.num_items)
            loss = loss + reg_loss

        return loss

    @torch.no_grad()
    def predict(self, users: torch.LongTensor, items: torch.LongTensor):
        """
        Predict scores for user-item pairs.

        Args:
            users: User IDs
            items: Item IDs

        Returns:
            Predicted scores
        """
        user_emb = self.user_embeddings(users)
        item_emb = self.item_embeddings(items)
        scores = (user_emb * item_emb).sum(dim=1)
        return scores

    @torch.no_grad()
    def recommend(self, user_ids: np.ndarray, k: int = 10, device: str = "cpu"):
        """
        Generate top-K recommendations for users (all-items ranking).

        Args:
            user_ids: Array of user IDs
            k: Number of recommendations
            device: Device to run on

        Returns:
            Array of shape (len(user_ids), k) with top-K item indices
        """
        self.eval()

        user_ids = torch.LongTensor(user_ids).to(device)
        user_emb = self.user_embeddings(user_ids)  # (num_users, embedding_dim)
        item_emb = self.item_embeddings.weight  # (num_items, embedding_dim)

        # Batch matrix multiplication for efficiency
        scores = torch.mm(user_emb, item_emb.t())  # (num_users, num_items)

        # Get top-K items
        _, top_items = torch.topk(scores, k, dim=1)

        return top_items.cpu().numpy()

    @torch.no_grad()
    def recommend_batch_candidates(
        self,
        users: torch.LongTensor,
        candidates: torch.LongTensor,
        k: int = 10,
    ):
        """
        Rank candidate items for users (candidate ranking for evaluation).

        Args:
            users: User IDs (batch_size,)
            candidates: Candidate item IDs (batch_size, num_candidates)
            k: Number of top items to return

        Returns:
            Top-K item indices within candidates
        """
        self.eval()

        batch_size, num_candidates = candidates.shape

        # Get user embeddings
        user_emb = self.user_embeddings(users)  # (batch_size, embedding_dim)

        # Get candidate embeddings
        cand_emb = self.item_embeddings(candidates)  # (batch_size, num_candidates, embedding_dim)

        # Compute scores: (batch_size, num_candidates)
        scores = (user_emb.unsqueeze(1) * cand_emb).sum(dim=2)

        # Get top-K
        _, top_idx = torch.topk(scores, k, dim=1)

        return top_idx


def train_bpr(
    model: BPR_MF_PyTorch,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    verbose: bool = True,
):
    """
    Train BPR model for one epoch.

    Args:
        model: BPR model
        train_loader: DataLoader with training triplets
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
        # Move batch to device
        users, items_pos, items_neg = batch
        users = users.squeeze().to(device)
        items_pos = items_pos.squeeze().to(device)
        items_neg = items_neg.squeeze().to(device)

        # Forward pass
        pos_scores, neg_scores = model(users, items_pos, items_neg)

        # Compute loss
        loss = model.bpr_loss(pos_scores, neg_scores)

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


def evaluate_bpr(
    model: BPR_MF_PyTorch,
    eval_loader: DataLoader,
    device: str,
    k_values: list[int] = [5, 10, 20],
    verbose: bool = True,
):
    """
    Evaluate BPR model using candidate ranking.

    Args:
        model: BPR model
        eval_loader: DataLoader with evaluation samples
        device: Device to evaluate on
        k_values: List of K values for metrics@K
        verbose: If True, show progress bar

    Returns:
        Dictionary with precision@K, recall@K, ndcg@K for each K
    """
    model.eval()

    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    iterator = tqdm(eval_loader, desc="Evaluating") if verbose else eval_loader

    with torch.no_grad():
        for batch in iterator:
            users, candidates, gt_positions = batch
            users = users.squeeze().to(device)
            candidates = candidates.to(device)
            gt_positions = gt_positions.squeeze()

            # Rank candidates
            top_k_max = max(k_values)
            top_idx = model.recommend_batch_candidates(users, candidates, k=top_k_max)

            # Convert to CPU for metric computation
            top_idx = top_idx.cpu().numpy()
            gt_positions = gt_positions.numpy()

            # Compute metrics for each sample
            for i in range(len(users)):
                gt_pos = gt_positions[i]
                ranking = top_idx[i]

                # Check if ground truth is in top-K for each K
                for k in k_values:
                    ranking_k = ranking[:k]
                    hit = 1 if gt_pos in ranking_k else 0

                    # Precision@K and Recall@K (same for single positive item)
                    metrics[f"precision@{k}"].append(hit)
                    metrics[f"recall@{k}"].append(hit)

                    # NDCG@K
                    if hit:
                        # Find position in ranking
                        pos_in_ranking = np.where(ranking_k == gt_pos)[0][0]
                        dcg = 1.0 / np.log2(pos_in_ranking + 2)
                        idcg = 1.0 / np.log2(2)  # Ideal: best item at position 0
                        ndcg = dcg / idcg
                    else:
                        ndcg = 0.0

                    metrics[f"ndcg@{k}"].append(ndcg)

    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}

    return avg_metrics
