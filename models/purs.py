"""
PURS (Personalized Unexpected Recommender System) Model.

Implements the paper: "PURS: Personalized Unexpected Recommender System for Improving User Satisfaction"
Recsys 2020.

Architecture:
1. CTR Scoring: Self-Attentive GRU + MLP (similar to Deep Interest Network)
2. Unexpectedness: Multi-cluster distance metric (Mean Shift on user history embeddings)
3. Sub-Gaussian Activation: Controls unexpectedness contribution
4. Personalized Factor: User-specific unexpectedness preference via attention

Final Score = CTR_score + f(unexpectedness) + personalized_factor
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MeanShift
from typing import Dict, Tuple, Optional
from collections import defaultdict


class SelfAttention(nn.Module):
    """Self-attention layer for weighting historical items."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute attention weights over hidden states.

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) or None

        Returns:
            Context: (batch_size, hidden_dim) - attention-weighted sum of hidden states
        """
        # Compute attention scores: (batch_size, seq_len, 1)
        scores = self.attention(hidden_states)

        if mask is not None:
            # Apply mask: set padded positions to very negative value
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-1e9"))

        # Softmax: (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)

        # Weighted sum: (batch_size, hidden_dim)
        context = (weights * hidden_states).sum(dim=1)

        return context


class PURS(nn.Module):
    """
    PURS: Personalized Unexpected Recommender System.

    Combines personalization (CTR prediction via GRU+attention) with
    unexpectedness (distance to user interest clusters).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        gru_hidden_dim: int = 32,
        num_clusters: int = 10,
        unexpectedness_weight: float = 0.5,
        history_length: int = 10,
        dropout: float = 0.1,
    ):
        """
        Initialize PURS model.

        Args:
            num_users: Total number of users
            num_items: Total number of items
            embedding_dim: Dimension of item embeddings
            gru_hidden_dim: Hidden dimension of GRU
            num_clusters: Number of clusters for Mean Shift (K in paper)
            unexpectedness_weight: Scaling factor λ for unexpectedness component
            history_length: Maximum length of user history sequences
            dropout: Dropout rate
        """
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_clusters = num_clusters
        self.unexpectedness_weight = unexpectedness_weight
        self.history_length = history_length

        # ============= Embeddings =============
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # ============= CTR Scoring Module =============
        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=dropout if gru_hidden_dim > 1 else 0,
        )

        # Self-attention on GRU outputs
        self.attention = SelfAttention(gru_hidden_dim)

        # MLP for CTR prediction
        # Input: [attention_output ⊕ item_embedding]
        mlp_input_dim = gru_hidden_dim + embedding_dim
        self.ctr_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # CTR in [0, 1]
        )

        # ============= Personalized Unexpectedness Factor =============
        # Self-Attentive MLP
        self.unexp_perception_mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Personalized factor in [0, 1]
        )

        # Store item embeddings for clustering (updated during training)
        self.item_embeddings_cached = None
        self.clusters_cached = None  # Will store cluster centroids per user
        self.cluster_sizes_cached = None

    def compute_user_clusters(self, user_history: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Mean Shift clusters for user's historical items.

        Args:
            user_history: (seq_len,) tensor of item indices, 0-padded

        Returns:
            Tuple of (centroids, cluster_sizes)
            - centroids: (num_clusters, embedding_dim) array
            - cluster_sizes: (num_clusters,) array of cluster sizes
        """
        # Remove padding (0 is pad value)
        valid_items = user_history[user_history > 0].cpu().numpy()

        if len(valid_items) == 0:
            # Empty history: return single "center" cluster
            return np.array([[0.0] * self.embedding_dim]), np.array([1])

        # Get embeddings for valid items
        valid_items_tensor = torch.LongTensor(valid_items).to(self.item_embedding.weight.device)
        history_embeddings = self.item_embedding(valid_items_tensor).detach().cpu().numpy()

        # Mean Shift clustering
        if len(valid_items) > 1:
            try:
                ms = MeanShift(bandwidth=None)  # Auto bandwidth estimation
                ms.fit(history_embeddings)
                centroids = ms.cluster_centers_
                labels = ms.labels_

                # Compute cluster sizes
                unique_labels = np.unique(labels)
                cluster_sizes = np.array([np.sum(labels == label) for label in unique_labels])

                return centroids, cluster_sizes
            except Exception as e:
                # Fallback: single cluster (mean of all items)
                print(f"Mean Shift clustering failed: {e}. Using single cluster.")
                return np.Mean(history_embeddings, axis=0, keepdims=True), np.array([len(valid_items)])
        else:
            # Single item: single cluster
            return history_embeddings, np.array([1])

    def compute_unexpectedness(self, candidate_embedding: np.ndarray, centroids: np.ndarray, cluster_sizes: np.ndarray) -> float:
        """
        Compute unexpectedness as distance to user interest clusters.

        Formula (from paper):
        unexpectedness = Σ_k distance(candidate, centroid_k) * (size_k / total_items)

        Args:
            candidate_embedding: (embedding_dim,) embedding of candidate item
            centroids: (num_clusters, embedding_dim) cluster centroids
            cluster_sizes: (num_clusters,) cluster sizes

        Returns:
            unexpectedness score in [0, 1]
        """
        if len(centroids) == 0:
            return 0.5  # Default unexpectedness

        # Compute L2 distances to all centroids
        distances = np.linalg.norm(candidate_embedding - centroids, axis=1)  # (num_clusters,)

        # Normalize distances to [0, 1]
        max_distance = np.max(distances) + 1e-9
        distances_normalized = distances / max_distance

        # Weighted sum by cluster size
        total_size = np.sum(cluster_sizes)
        weights = cluster_sizes / total_size
        unexpectedness = np.sum(distances_normalized * weights)

        # Clip to [0, 1]
        return float(np.clip(unexpectedness, 0.0, 1.0))

    @staticmethod
    def sub_gaussian_activation(x: torch.Tensor) -> torch.Tensor:
        """
        Sub-Gaussian activation function: f(x) = x * exp(-x)

        Properties:
        - Smooth and continuous
        - Peaks around x=1
        - Rapid decay at high values (prevents over-rewarding extreme unexpectedness)
        - Output range: [0, 1/e ≈ 0.368]

        Args:
            x: Tensor of unexpectedness values in [0, 1]

        Returns:
            Activated values
        """
        return x * torch.exp(-x)

    def forward_ctr(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor, histories: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTR scores using attention-based GRU.

        Args:
            user_ids: (batch_size,) user indices
            item_ids: (batch_size,) item indices
            histories: (batch_size, history_length) padded item sequences

        Returns:
            ctr_scores: (batch_size,) CTR predictions in [0, 1]
        """
        batch_size = user_ids.shape[0]

        # Get embeddings
        history_embeddings = self.item_embedding(histories)  # (batch_size, history_length, embedding_dim)
        item_embeddings = self.item_embedding(item_ids)  # (batch_size, embedding_dim)

        # Create mask for padded positions (0 is pad)
        mask = (histories > 0).float()  # (batch_size, history_length)

        # GRU forward
        gru_output, _ = self.gru(history_embeddings)  # (batch_size, history_length, gru_hidden_dim)

        # Self-attention on GRU output
        context = self.attention(gru_output, mask)  # (batch_size, gru_hidden_dim)

        # Concatenate context and item embedding
        combined = torch.cat([context, item_embeddings], dim=1)  # (batch_size, gru_hidden_dim + embedding_dim)

        # MLP for CTR prediction
        ctr_scores = self.ctr_mlp(combined)  # (batch_size, 1)

        return ctr_scores.squeeze(1)  # (batch_size,)

    def forward_unexpectedness_perception(self, histories: torch.Tensor) -> torch.Tensor:
        """
        Compute personalized unexpectedness preference factor.

        Args:
            histories: (batch_size, history_length) padded item sequences

        Returns:
            unexp_factors: (batch_size,) personalized preference in [0, 1]
        """
        # Get embeddings
        history_embeddings = self.item_embedding(histories)  # (batch_size, history_length, embedding_dim)

        # Create mask for padded positions
        mask = (histories > 0).float()  # (batch_size, history_length)

        # GRU forward
        gru_output, _ = self.gru(history_embeddings)  # (batch_size, history_length, gru_hidden_dim)

        # Self-attention
        context = self.attention(gru_output, mask)  # (batch_size, gru_hidden_dim)

        # MLP for unexpectedness perception
        unexp_factors = self.unexp_perception_mlp(context)  # (batch_size, 1)

        return unexp_factors.squeeze(1)  # (batch_size,)

    def forward(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor, histories: torch.Tensor, compute_unexpectedness: bool = True
    ) -> torch.Tensor:
        """
        Compute unified PURS scores.

        Args:
            user_ids: (batch_size,) user indices
            item_ids: (batch_size,) item indices
            histories: (batch_size, history_length) padded item sequences
            compute_unexpectedness: If True, compute full score with unexpectedness

        Returns:
            scores: (batch_size,) unified PURS scores
        """
        batch_size = user_ids.shape[0]

        # CTR score
        ctr_scores = self.forward_ctr(user_ids, item_ids, histories)  # (batch_size,)

        if not compute_unexpectedness:
            return ctr_scores

        # Unexpectedness component (computed per user)
        unexpectedness_scores = []
        for i in range(batch_size):
            user_history = histories[i]
            candidate_item = item_ids[i].item()

            # Compute clusters for this user
            centroids, cluster_sizes = self.compute_user_clusters(user_history)

            # Get candidate embedding
            candidate_embedding = self.item_embedding(torch.LongTensor([candidate_item]).to(item_ids.device)).detach().cpu().numpy()[0]

            # Compute unexpectedness
            unexp = self.compute_unexpectedness(candidate_embedding, centroids, cluster_sizes)
            unexpectedness_scores.append(unexp)

        unexpectedness_scores = torch.from_numpy(np.array(unexpectedness_scores)).float().to(item_ids.device)

        # Apply sub-Gaussian activation
        unexp_activated = self.sub_gaussian_activation(unexpectedness_scores)

        # Personalized factor
        unexp_factors = self.forward_unexpectedness_perception(histories)

        # Combine: Final Score = CTR + λ * f(unexpectedness) + personalized_factor
        final_scores = ctr_scores + self.unexpectedness_weight * unexp_activated + unexp_factors

        return final_scores

    def recommend(
        self,
        user_ids: np.ndarray,
        all_items: np.ndarray = None,
        histories: torch.Tensor = None,
        k: int = 10,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Generate top-K recommendations for users.

        Args:
            user_ids: (num_users,) array of user indices
            all_items: (num_items,) array of all item indices or None for [0, 1, ..., num_items-1]
            histories: (num_users, history_length) tensor of user histories
            k: Number of top-K items to return
            device: Device to use for computation

        Returns:
            (num_users, k) array of recommended item indices
        """
        if all_items is None:
            all_items = np.arange(self.num_items)

        num_users = len(user_ids)
        num_items = len(all_items)

        # Prepare tensors
        user_ids_tensor = torch.LongTensor(user_ids).to(device)
        all_items_tensor = torch.LongTensor(all_items).to(device)

        recommendations = []

        with torch.no_grad():
            for i, user_id in enumerate(user_ids):
                # Get user history
                if histories is not None:
                    user_history = histories[i : i + 1].to(device)  # (1, history_length)
                else:
                    user_history = torch.zeros(1, self.history_length, dtype=torch.long).to(device)

                # Compute scores for all items
                user_ids_batch = torch.LongTensor([user_id]).to(device)
                scores = []

                for item_id in all_items:
                    item_id_batch = torch.LongTensor([item_id]).to(device)
                    score = self.forward(user_ids_batch, item_id_batch, user_history, compute_unexpectedness=True)
                    scores.append(score.item())

                scores = np.array(scores)

                # Get top-K
                top_indices = np.argsort(-scores)[:k]
                top_items = all_items[top_indices]

                recommendations.append(top_items)

        return np.array(recommendations)
