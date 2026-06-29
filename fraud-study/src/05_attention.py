"""
Single-head attention do zero com números pequenos.
Mostra a matriz de atenção e os pesos que emergem para um cliente de fraude.

scores = softmax(Q·Kᵀ / √d_k)
output = scores · V

Exporta: def attention(Q, K, V, mask=None) → (output, weights)
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from config import SEED, D_MODEL, SEQ_LEN, DATA_PATH
from config import TIPOS_INV, PAISES_INV, DEVICES_INV

torch.manual_seed(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")


# ── Função pública ─────────────────────────────────────────────────────────────
def attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.
    Q, K, V: [..., seq_len, d_k]
    mask:    [..., seq_len] — 1=real, 0=padding
    Retorna: (output [..., seq_len, d_k], weights [..., seq_len, seq_len])
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Zerar atenção para posições de padding
        mask_2d = mask.unsqueeze(-2)  # [..., 1, seq_len]
        scores = scores.masked_fill(mask_2d == 0, -1e9)

    weights = F.softmax(scores, dim=-1)
    output  = torch.matmul(weights, V)
    return output, weights


# ── Main standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import importlib
    _tok = importlib.import_module("src.01_tokenizer")
    _emb = importlib.import_module("src.02_embeddings")
    _fus = importlib.import_module("src.03_local_fusion")
    _pos = importlib.import_module("src.04_positional")

    load_and_tokenize = _tok.load_and_tokenize
    FieldEncoders     = _emb.FieldEncoders
    LocalFusion       = _fus.LocalFusion
    PositionalEncoding = _pos.PositionalEncoding

    # ── Carregar dados e pegar cliente com fraude ──────────────────────────────
    dados = load_and_tokenize()
    X, y, masks = dados["X"], dados["y"], dados["masks"]

    import numpy as np
    cid = int(np.where(y == 1)[0][0])

    bloco(1, f"Sequência do cliente {cid} (fraude=1)")
    print()
    x_cliente = torch.tensor(X[cid:cid+1])   # [1, SEQ_LEN, 5]
    m_cliente  = torch.tensor(masks[cid:cid+1])  # [1, SEQ_LEN]

    for pos in range(SEQ_LEN):
        m = int(masks[cid, pos])
        if m == 0:
            print(f"  pos {pos}: padding")
        else:
            tipo   = TIPOS_INV.get(int(X[cid, pos, 0]), "?")
            pais   = PAISES_INV.get(int(X[cid, pos, 1]), "?")
            device = DEVICES_INV.get(int(X[cid, pos, 2]), "?")
            hora_norm = X[cid, pos, 4]
            suspeito = "🚨" if hora_norm < 0.26 else ("⚠️ " if tipo in ("troca_senha", "compra_negada") else "  ")
            print(f"  pos {pos}: {tipo:<16} {pais:<4} {device:<26} {suspeito}")

    # ── Construir representação via pipeline ───────────────────────────────────
    encoders = FieldEncoders()
    fusion   = LocalFusion()
    pos_enc  = PositionalEncoding()

    with torch.no_grad():
        embs   = encoders(x_cliente)
        tokens = fusion(embs)           # [1, SEQ_LEN, D_MODEL]
        tokens = pos_enc(tokens)

    # Projeções Q, K, V aleatórias (não treinadas — para fins didáticos)
    torch.manual_seed(SEED)
    d_k = D_MODEL
    Wq = torch.randn(D_MODEL, d_k) * 0.1
    Wk = torch.randn(D_MODEL, d_k) * 0.1
    Wv = torch.randn(D_MODEL, d_k) * 0.1

    Q = tokens @ Wq   # [1, SEQ_LEN, d_k]
    K = tokens @ Wk
    V = tokens @ Wv

    output, weights = attention(Q, K, V, mask=m_cliente)

    bloco(2, "Matriz de atenção — último evento olhando para todos")
    ultimo_pos = int(m_cliente[0].nonzero()[-1])
    pesos_ultimo = weights[0, ultimo_pos].detach().tolist()

    print(f"\n  Evento na pos {ultimo_pos} (último evento real) olhando para trás:\n")
    print(f"  {'pos':<5} {'peso':>6}   {'barra'}")
    print(f"  {'─'*45}")
    for pos in range(SEQ_LEN):
        peso = pesos_ultimo[pos]
        barra = "█" * int(peso * 80)
        m_str = "(padding)" if masks[cid, pos] == 0 else ""
        print(f"  {pos:<5} {peso:>6.3f}   {barra}  {m_str}")

    # ── Interpretação ──────────────────────────────────────────────────────────
    bloco(3, "Interpretação dos pesos")
    pares = sorted(enumerate(pesos_ultimo), key=lambda x: x[1], reverse=True)
    print()
    for pos, peso in pares[:4]:
        if masks[cid, pos] == 0:
            continue
        tipo = TIPOS_INV.get(int(X[cid, pos, 0]), "?")
        pais = PAISES_INV.get(int(X[cid, pos, 1]), "?")
        print(f"  pos {pos} ({tipo} / {pais}): {peso*100:.1f}% da atenção")
    print()
    print("  Pesos de atenção = o quanto cada evento passado influencia a decisão.")
    print("  Eventos com padrão suspeito (troca_senha, país estrangeiro, madrugada)")
    print("  recebem mais atenção do modelo após o treino.")
