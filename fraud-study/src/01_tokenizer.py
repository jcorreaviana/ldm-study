"""
Converte transactions.csv em arrays numpy prontos para o modelo.
Pré-processamento puro — sem embeddings ainda.

Exporta: load_and_tokenize(csv_path) → dict com X, y, masks
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from config import (
    SEED, SEQ_LEN, N_CLIENTES, TIPOS, PAISES, DEVICES,
    DATA_PATH, TIPOS_INV, PAISES_INV, DEVICES_INV,
)

np.random.seed(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")

def ok(msg, start=None):
    elapsed = f"  ({time.time()-start:.1f}s)" if start else ""
    print(f"✅ {msg}{elapsed}")

def info(msg):
    print(f"ℹ️  {msg}")


# ── Função pública (importável pelo 09) ────────────────────────────────────────
def load_and_tokenize(csv_path=DATA_PATH):
    """
    Lê o CSV e retorna:
      X     : [N_CLIENTES, SEQ_LEN, 5]   — (tipo, pais, device, valor_norm, hora_norm)
      y     : [N_CLIENTES]               — 0/1
      masks : [N_CLIENTES, SEQ_LEN]      — 1=evento real, 0=padding
    """
    if not os.path.exists(csv_path):
        print("❌ Dados não encontrados. Execute primeiro: python data/generate_data.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    n_clientes = df['cliente_id'].nunique()
    X     = np.zeros((n_clientes, SEQ_LEN, 5), dtype=np.float32)
    y     = np.zeros(n_clientes, dtype=np.float32)
    masks = np.zeros((n_clientes, SEQ_LEN), dtype=np.float32)

    for cid, grupo in df.groupby('cliente_id'):
        eventos = grupo.sort_values('evento_num')
        label   = int(eventos['fraude'].iloc[0])
        y[cid]  = label

        # Pegar os últimos SEQ_LEN eventos (mais recentes)
        if len(eventos) > SEQ_LEN:
            eventos = eventos.tail(SEQ_LEN)

        n = len(eventos)
        # Left-padding: preencher da posição (SEQ_LEN - n) em diante
        offset = SEQ_LEN - n

        for i, (_, row) in enumerate(eventos.iterrows()):
            pos = offset + i
            X[cid, pos, 0] = TIPOS.get(row['tipo'], 0)
            X[cid, pos, 1] = PAISES.get(row['pais'], 0)
            X[cid, pos, 2] = DEVICES.get(row['device'], 0)
            X[cid, pos, 3] = min(float(row['valor']) / 10000.0, 1.0)
            X[cid, pos, 4] = float(row['hora']) / 23.0
            masks[cid, pos] = 1.0

    return {"X": X, "y": y, "masks": masks}


# ── Main standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = time.time()
    bloco(1, "Carregando e tokenizando transações")

    dados = load_and_tokenize()
    X, y, masks = dados["X"], dados["y"], dados["masks"]

    ok("Tokenização concluída", t)
    print()
    print(f"  Shape X:     {X.shape}   ← [clientes, eventos, campos]")
    print(f"  Shape y:     {y.shape}          ← [clientes]")
    print(f"  Shape masks: {masks.shape}   ← [clientes, eventos]")
    n_fraude = int(y.sum())
    print(f"  Fraude:      {n_fraude} ({n_fraude/len(y)*100:.1f}%)")

    # ── Exemplo: cliente com fraude ────────────────────────────────────────────
    bloco(2, "Exemplo: sequência de um cliente com fraude")
    # Pegar primeiro cliente com fraude
    cid_fraude = int(np.where(y == 1)[0][0])
    print(f"\n  Cliente {cid_fraude} (fraude=1)")
    print(f"  {'pos':<4} {'tipo':<16} {'pais':<6} {'device':<26} {'valor':>7} {'hora':>6}  {'mask'}")
    print(f"  {'─'*75}")

    for pos in range(SEQ_LEN):
        m = int(masks[cid_fraude, pos])
        if m == 0:
            tipo_str   = "padding"
            pais_str   = "─"
            device_str = "─"
            val_str    = "0.000"
            hora_str   = "0.00"
            nota       = "← padding"
        else:
            tipo_str   = TIPOS_INV.get(int(X[cid_fraude, pos, 0]), "?")
            pais_str   = PAISES_INV.get(int(X[cid_fraude, pos, 1]), "?")
            device_str = DEVICES_INV.get(int(X[cid_fraude, pos, 2]), "?")
            val_str    = f"{X[cid_fraude, pos, 3]:.3f}"
            hora_str   = f"{X[cid_fraude, pos, 4]:.2f}"
            if X[cid_fraude, pos, 4] < 0.26:  # hora < 6
                nota = "🚨 suspeito"
            elif tipo_str in ("troca_senha", "compra_negada"):
                nota = "⚠️  alerta"
            else:
                nota = ""
        print(f"  {pos:<4} {tipo_str:<16} {pais_str:<6} {device_str:<26} {val_str:>7} {hora_str:>6}  {nota}")

    # ── Exemplo: cliente normal ────────────────────────────────────────────────
    bloco(3, "Exemplo: sequência de um cliente normal")
    cid_normal = int(np.where(y == 0)[0][0])
    print(f"\n  Cliente {cid_normal} (fraude=0)")
    print(f"  {'pos':<4} {'tipo':<16} {'pais':<6} {'device':<26} {'valor':>7} {'hora':>6}")
    print(f"  {'─'*65}")
    for pos in range(SEQ_LEN):
        m = int(masks[cid_normal, pos])
        if m == 0:
            print(f"  {pos:<4} {'padding':<16} {'─':<6} {'─':<26} {'0.000':>7} {'0.00':>6}")
        else:
            tipo_str   = TIPOS_INV.get(int(X[cid_normal, pos, 0]), "?")
            pais_str   = PAISES_INV.get(int(X[cid_normal, pos, 1]), "?")
            device_str = DEVICES_INV.get(int(X[cid_normal, pos, 2]), "?")
            val_str    = f"{X[cid_normal, pos, 3]:.3f}"
            hora_str   = f"{X[cid_normal, pos, 4]:.2f}"
            print(f"  {pos:<4} {tipo_str:<16} {pais_str:<6} {device_str:<26} {val_str:>7} {hora_str:>6}")
