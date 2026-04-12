"""
SASRec model for sequential recommendation.
"""

import torch
import torch.nn as nn


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.

    This implementation uses a Transformer encoder over item histories and
    predicts next-item preference via dot-product scoring.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_length: int = 50,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
        ffn_dim: int = 256,
        reg_lambda: float = 0.0,
    ):
        super().__init__()

        self.num_items = num_items
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.pad_token = num_items  # reserve one extra id for sequence padding
        self.reg_lambda = reg_lambda

        self.item_embedding = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=embedding_dim,
            padding_idx=self.pad_token,
        )
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def _encode_history(self, histories: torch.LongTensor) -> torch.Tensor:
        """Encode interaction histories into a context vector per user."""
        batch_size, seq_len = histories.shape

        positions = torch.arange(seq_len, device=histories.device).unsqueeze(0).expand(batch_size, -1)
        x = self.item_embedding(histories) + self.position_embedding(positions)
        x = self.dropout(self.layer_norm(x))

        # Prevent attending to future positions for autoregressive sequence modeling.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=histories.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = histories.eq(self.pad_token)

        encoded = self.encoder(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        valid_lengths = (~padding_mask).long().sum(dim=1).clamp(min=1)
        last_indices = valid_lengths - 1
        context = encoded[torch.arange(batch_size, device=histories.device), last_indices]
        return context

    def score_items(self, histories: torch.LongTensor, items: torch.LongTensor) -> torch.Tensor:
        """
        Score candidate items given user history.

        Args:
            histories: (batch_size, seq_len)
            items: (batch_size,) or (batch_size, num_candidates)

        Returns:
            Scores with shape (batch_size,) or (batch_size, num_candidates)
        """
        context = self._encode_history(histories)

        if items.dim() == 1:
            item_emb = self.item_embedding(items)
            return (context * item_emb).sum(dim=1)

        if items.dim() == 2:
            item_emb = self.item_embedding(items)
            return torch.einsum("bd,bcd->bc", context, item_emb)

        raise ValueError(f"Expected items to be rank 1 or 2, got shape={tuple(items.shape)}")

    def forward(self, histories: torch.LongTensor, items: torch.LongTensor) -> torch.Tensor:
        return self.score_items(histories, items)
