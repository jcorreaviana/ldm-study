"""
Transformer tabular com atenção entre features (estilo FT-Transformer) para detecção de fraude.

Arquitetura (ver arquitetura.md para justificativas e diagrama):
  30 features numéricas (V1-V28 + Amount_z + Time_z)
    -> Feature Tokenizer (1 embedding por feature, projeção linear escalar->vetor)
    -> + token [CLS]
    -> N=3 blocos de Transformer encoder (pre-norm), h=4 cabeças de atenção
    -> token [CLS] final -> cabeça de classificação -> logit de fraude

Não treina nada neste arquivo — só define o modelo e valida shapes/params.
"""
import torch
import torch.nn as nn
import math


class FeatureTokenizer(nn.Module):
    """Cada feature numérica escalar vira um token (vetor de dimensão d_model).
    token_i = x_i * W_i + b_i   (W_i, b_i aprendidos, um por feature)
    """
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.uniform_(self.weight, -1 / math.sqrt(d_model), 1 / math.sqrt(d_model))
        nn.init.uniform_(self.bias, -1 / math.sqrt(d_model), 1 / math.sqrt(d_model))

    def forward(self, x):
        # x: (batch, n_features) -> tokens: (batch, n_features, d_model)
        return x.unsqueeze(-1) * self.weight + self.bias


class TransformerEncoderBlock(nn.Module):
    """Bloco pre-norm: atenção multi-cabeça entre features + FFN, com residual."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        # atenção entre os tokens de features (não entre transações)
        h = self.norm1(x)
        attn_out, attn_weights = self.attn(h, h, h, need_weights=return_attn, average_attn_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return (x, attn_weights) if return_attn else (x, None)


class FraudTabularTransformer(nn.Module):
    def __init__(self, n_features: int = 30, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, d_ff: int = 128, dropout: float = 0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, return_attn=False):
        batch_size = x.size(0)
        tokens = self.tokenizer(x)                                   # (B, n_features, d_model)
        cls = self.cls_token.expand(batch_size, -1, -1)               # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)                         # (B, n_features+1, d_model)

        attn_maps = []
        for block in self.blocks:
            seq, attn_w = block(seq, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn_w)

        seq = self.final_norm(seq)
        cls_out = seq[:, 0, :]                                        # representação final do [CLS]
        logit = self.classifier(cls_out).squeeze(-1)                  # (B,)
        return (logit, attn_maps) if return_attn else logit


def weighted_bce_loss(logits, targets, class_weight: dict, cluster_boost_mask=None, cluster_boost_factor: float = 2.0):
    """BCE ponderada: peso base por classe (desbalanceamento) x boost extra para
    fraudes do cluster 'majoritário/brando' (o mais parecido com transação legítima).

    cluster_boost_mask: tensor booleano (B,) = True para exemplos de fraude (y=1)
    que pertencem ao cluster prioritário. None se não disponível no batch.
    """
    base_bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    weight = torch.where(targets == 1, class_weight[1], class_weight[0])
    if cluster_boost_mask is not None:
        weight = torch.where(cluster_boost_mask, weight * cluster_boost_factor, weight)
    return (base_bce * weight).mean()


if __name__ == '__main__':
    torch.manual_seed(42)
    model = FraudTabularTransformer(n_features=30, d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros totais: {n_params:,}")
    print("\nParâmetros por módulo:")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:15s}: {n:,}")

    # sanity check com dado real do treino
    import pandas as pd
    train = pd.read_csv('/sessions/magical-busy-rubin/mnt/outputs/tt-docs/projetos/fraude/preprocessing/train.csv')
    v_cols = [c for c in train.columns if c.startswith('V')]
    feature_cols = v_cols + ['Amount_z', 'Time_z']
    batch = torch.tensor(train[feature_cols].head(8).values, dtype=torch.float32)
    y = torch.tensor(train['Class'].head(8).values, dtype=torch.float32)

    logits, attn_maps = model(batch, return_attn=True)
    print(f"\nInput shape:  {tuple(batch.shape)}")
    print(f"Output shape (logits): {tuple(logits.shape)}")
    print(f"Nº de mapas de atenção (1 por camada): {len(attn_maps)}")
    print(f"Shape de cada mapa de atenção (batch, heads, seq, seq): {tuple(attn_maps[0].shape)}")

    class_weight = {0: torch.tensor(0.5008), 1: torch.tensor(300.0121)}
    loss = weighted_bce_loss(logits, y, class_weight)
    print(f"\nLoss (batch de teste, pesos ok): {loss.item():.4f}")
