"""
Multi-head attention: N_HEADS cabeças em paralelo, cada uma com projeções independentes.
Cada cabeça aprende um tipo diferente de relação na sequência.

Exporta: class MultiHeadAttention(nn.Module)
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import importlib
from config import SEED, D_MODEL, N_HEADS

_att = importlib.import_module("src.05_attention")
attention = _att.attention

torch.manual_seed(SEED)

assert D_MODEL % N_HEADS == 0, "D_MODEL deve ser divisível por N_HEADS"
D_HEAD = D_MODEL // N_HEADS  # dimensão por cabeça

# ── Helpers ────────────────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")


# ── Classe pública ─────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    N_HEADS cabeças de atenção independentes em paralelo.
    Entrada:  [batch, seq_len, D_MODEL]
    Saída:    [batch, seq_len, D_MODEL]
    """
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # Uma projeção linear Q/K/V por cabeça
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)  # projeção de saída

    def forward(self, x, mask=None):
        """
        x:    [batch, seq_len, D_MODEL]
        mask: [batch, seq_len]  — 1=real, 0=padding
        """
        B, S, _ = x.shape

        # Projetar e dividir em cabeças: [batch, n_heads, seq_len, d_head]
        def split_heads(tensor):
            return tensor.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.Wq(x))
        K = split_heads(self.Wk(x))
        V = split_heads(self.Wv(x))

        # Mask ajustada para [batch, 1, seq_len] — attention() adiciona o unsqueeze restante
        mask_heads = mask.unsqueeze(1) if mask is not None else None

        output, weights = attention(Q, K, V, mask=mask_heads)

        # Reunir cabeças: [batch, seq_len, D_MODEL]
        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.Wo(output), weights  # weights: [batch, n_heads, seq_len, seq_len]


# ── Main standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _tok = importlib.import_module("src.01_tokenizer")
    _emb = importlib.import_module("src.02_embeddings")
    _fus = importlib.import_module("src.03_local_fusion")
    _pos = importlib.import_module("src.04_positional")

    import numpy as np
    dados  = _tok.load_and_tokenize()
    X, y, masks = dados["X"], dados["y"], dados["masks"]

    cid = int(np.where(y == 1)[0][0])
    x_t = torch.tensor(X[cid:cid+1])
    m_t = torch.tensor(masks[cid:cid+1])

    # Pipeline até tokens posicionais
    encoders = _emb.FieldEncoders()
    fusion   = _fus.LocalFusion()
    pos_enc  = _pos.PositionalEncoding()

    with torch.no_grad():
        tokens = pos_enc(fusion(encoders(x_t)))

    mha = MultiHeadAttention()
    mha.eval()

    with torch.no_grad():
        output, weights = mha(tokens, mask=m_t)

    # Posição do último evento real
    ultimo_pos = int(m_t[0].nonzero()[-1])

    bloco(1, f"4 cabeças de atenção — cliente {cid} (fraude=1)")
    print(f"\n  Evento analisado: pos {ultimo_pos} (último evento real)")
    print(f"  Cada cabeça aprende um padrão diferente na mesma sequência.\n")

    nomes_cabeca = [
        "padrão temporal",
        "padrão geográfico",
        "padrão de valor",
        "padrão de dispositivo",
    ]

    from config import TIPOS_INV, PAISES_INV
    for h in range(N_HEADS):
        pesos = weights[0, h, ultimo_pos].detach().tolist()
        top2  = sorted(enumerate(pesos), key=lambda x: x[1], reverse=True)[:2]
        top2  = [(pos, p) for pos, p in top2 if masks[cid, pos] > 0]

        print(f"  Cabeça {h+1} — {nomes_cabeca[h]}:")
        for pos, peso in top2:
            tipo = TIPOS_INV.get(int(X[cid, pos, 0]), "?")
            pais = PAISES_INV.get(int(X[cid, pos, 1]), "?")
            print(f"    maior atenção: pos{pos}={peso:.2f}  ({tipo} / {pais})")
        print()

    bloco(2, "Shape do output")
    print(f"\n  Input:  {tokens.shape}")
    print(f"  Output: {output.shape}")
    print(f"\n  A dimensionalidade se mantém — {D_MODEL} dims por evento.")
    print(f"  O conteúdo muda: agora cada token 'ouviu' todos os outros.")

    bloco(3, "Parâmetros")
    total = sum(p.numel() for p in mha.parameters())
    print(f"\n  Wq + Wk + Wv + Wo: {total} parâmetros")
    print(f"  D_HEAD por cabeça: {D_MODEL} ÷ {N_HEADS} = {D_HEAD} dims")
    print(f"  {N_HEADS} cabeças × {D_HEAD} dims = {D_MODEL} dims total (recombinados em Wo)")
