"""
Temporal backbone completo: FieldEncoders + LocalFusion + PositionalEncoding
+ N camadas de (MultiHeadAttention + FeedForward + LayerNorm).

Exporta: class FraudTransformer(nn.Module)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import importlib
from config import SEED, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, SEQ_LEN

_emb = importlib.import_module("src.02_embeddings")
_fus = importlib.import_module("src.03_local_fusion")
_pos = importlib.import_module("src.04_positional")
_mha = importlib.import_module("src.06_multihead")

FieldEncoders      = _emb.FieldEncoders
LocalFusion        = _fus.LocalFusion
PositionalEncoding = _pos.PositionalEncoding
MultiHeadAttention = _mha.MultiHeadAttention

torch.manual_seed(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")


# ── Camada do encoder ──────────────────────────────────────────────────────────
class TransformerEncoderLayer(nn.Module):
    """
    Uma camada do transformer:
      1. Multi-head attention + residual + LayerNorm
      2. Feed-forward (Linear → ReLU → Linear) + residual + LayerNorm
    """
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sublayer 1: atenção com residual
        att_out, _ = self.attention(x, mask=mask)
        x = self.norm1(x + self.dropout(att_out))

        # Sublayer 2: feed-forward com residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# ── Classe pública ─────────────────────────────────────────────────────────────
class FraudTransformer(nn.Module):
    """
    Backbone temporal completo.
    Entrada:  x [batch, seq_len, 5], mask [batch, seq_len]
    Saída:    [batch, seq_len, D_MODEL]  — representação contextualizada de cada evento
    """
    def __init__(self):
        super().__init__()
        self.field_encoders = FieldEncoders()
        self.local_fusion   = LocalFusion()
        self.pos_encoding   = PositionalEncoding()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer() for _ in range(N_LAYERS)
        ])
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x, mask=None):
        # 1. Field encoders: campos → vetores
        embs = self.field_encoders(x)
        # 2. Local fusion: 5 vetores → 1 token por evento
        h = self.local_fusion(embs)
        # 3. Positional encoding: soma informação de ordem
        h = self.pos_encoding(h)
        # 4. N camadas de atenção + feed-forward
        for layer in self.layers:
            h = layer(h, mask=mask)
        return self.norm(h)


# ── Main standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = FraudTransformer()

    bloco(1, "Arquitetura")
    print(f"\n  FraudTransformer(")
    print(f"    field_encoders  : FieldEncoders(tipo=Emb(7,{D_MODEL}), pais=Emb(6,{D_MODEL}), device=Emb(4,{D_MODEL}), valor/hora=Linear)")
    print(f"    local_fusion    : LocalFusion(Linear({D_MODEL*5}→{D_MODEL}) + ReLU + LayerNorm)")
    print(f"    pos_encoding    : PositionalEncoding(d={D_MODEL}, max_len={SEQ_LEN})")
    for i in range(N_LAYERS):
        print(f"    layer[{i}]        : TransformerEncoderLayer(d={D_MODEL}, heads={N_HEADS}, ff={D_FF})")
    print(f"    norm            : LayerNorm({D_MODEL})")
    print(f"  )")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total de parâmetros: {total_params:,}  ← pequeno, roda em CPU")

    bloco(2, "Forward pass (1 cliente, 10 eventos)")
    _tok = importlib.import_module("src.01_tokenizer")
    import numpy as np
    dados = _tok.load_and_tokenize()
    X, y, masks = dados["X"], dados["y"], dados["masks"]

    cid = int(np.where(y == 1)[0][0])
    x_t = torch.tensor(X[cid:cid+1])
    m_t = torch.tensor(masks[cid:cid+1])

    model.eval()

    # Rastrear shapes em cada etapa
    with torch.no_grad():
        embs   = model.field_encoders(x_t)
        after_enc = torch.stack(list(embs.values()), dim=-1)  # só para visualização
        after_fus = model.local_fusion(embs)
        after_pos = model.pos_encoding(after_fus)
        h = after_pos
        shapes_layers = []
        for layer in model.layers:
            h = layer(h, mask=m_t)
            shapes_layers.append(h.shape)
        h = model.norm(h)

    print(f"\n  Input:          {x_t.shape}   ← [batch, seq_len, n_campos]")
    print(f"  After encoders: {after_fus.shape}  ← [batch, seq_len, D_MODEL]")
    print(f"  After fusion:   {after_fus.shape}  ← mesma shape, campos fundidos")
    print(f"  After pos enc:  {after_pos.shape}  ← + informação de posição")
    for i, s in enumerate(shapes_layers):
        print(f"  After layer {i+1}:  {s}  ← contexto entre eventos")
    print(f"  Output final:   {h.shape}  ← pronto para o classificador")
