"""
Transformer sequencial para detecção de card testing.

Cada transação (token) = embeddings de tipo/merchant + features numéricas
(valor, hora, saldo_antes, saldo_depois) projetadas para d_model. O encoder
usa a máscara "banded causal" (ver dataset.py) para que a representação de
cada posição só dependa dela mesma e das até `window_size - 1` transações
anteriores do mesmo cliente. A saída é uma probabilidade de fraude POR
POSICAO (uma para cada transação da sequência), não uma classificação única
da sequência inteira.
"""

import math

import torch
import torch.nn as nn

from utils import PAD_IDX


class SinusoidalPositionalEncoding(nn.Module):
    """Codificacao posicional classica (Vaswani et al.), sem parametros treinaveis."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerFraudDetector(nn.Module):
    def __init__(
        self,
        n_tipo: int,
        n_merchant: int,
        n_numeric: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        cat_emb_dim: int = 8,
        max_len: int = 64,
    ):
        super().__init__()
        # +1 porque o indice 0 e reservado para padding (ver utils.PAD_IDX)
        self.tipo_emb = nn.Embedding(n_tipo + 1, cat_emb_dim, padding_idx=PAD_IDX)
        self.merchant_emb = nn.Embedding(n_merchant + 1, cat_emb_dim, padding_idx=PAD_IDX)

        input_dim = n_numeric + 2 * cat_emb_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,        # (B, L, n_numeric)
        tipo_idx: torch.Tensor,       # (B, L)
        merchant_idx: torch.Tensor,   # (B, L)
        attn_mask: torch.Tensor,      # (L, L) aditiva, 0/-inf
    ) -> torch.Tensor:
        # Nota importante: não passamos src_key_padding_mask para o encoder,
        # de propósito. O padding sempre fica à direita de cada sequência
        # (right-padding, ver dataset.collate_fn) e attn_mask já é causal
        # (posição i só atende a j <= i). Logo nenhuma posição real (i menor
        # que o comprimento da sequência) jamais enxerga uma chave de
        # padding -- o key_padding_mask seria redundante para elas.
        #
        # Se aplicássemos key_padding_mask mesmo assim, as posições de QUERY
        # que são puro padding (i >= comprimento real) e cuja janela causal
        # não alcança nenhuma posição real ficariam com TODAS as chaves
        # bloqueadas (-inf em toda a linha), gerando softmax NaN. E NaN * 0
        # dentro do masked_bce_loss (em train.py) ainda resulta em NaN --
        # isso envenenaria o gradiente do batch inteiro mesmo mascarando a
        # loss depois. Como as posições de padding já são descartadas via
        # valid_mask na loss e nas métricas, é mais seguro deixá-las atender
        # (sem sentido, mas sem produzir NaN) apenas dentro da própria
        # janela causal.
        cat = torch.cat([self.tipo_emb(tipo_idx), self.merchant_emb(merchant_idx)], dim=-1)
        x = torch.cat([numeric, cat], dim=-1)
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        h = self.encoder(x, mask=attn_mask)
        logits = self.classifier(h).squeeze(-1)  # (B, L) -- logit por transacao
        return logits
