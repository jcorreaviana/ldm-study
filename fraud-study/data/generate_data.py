"""
Gera event stream sintético de transações bancárias com padrões realistas de fraude.
Saída: data/transactions.csv
"""
import sys
import os
import time
import random
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    SEED, N_CLIENTES, N_FRAUD, MAX_EVENTOS, MIN_EVENTOS,
    TIPOS, PAISES, DEVICES, DATA_PATH,
)

random.seed(SEED)
np.random.seed(SEED)

# ── Helpers de output ──────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")

def ok(msg, start=None):
    elapsed = f"  ({time.time()-start:.1f}s)" if start else ""
    print(f"✅ {msg}{elapsed}")

def info(msg):
    print(f"ℹ️  {msg}")


# ── Probabilidades por perfil ──────────────────────────────────────────────────
# Fase normal (todos os clientes)
PROB_TIPO_NORMAL = {
    "login": 0.30, "compra": 0.30, "consulta_saldo": 0.20,
    "transferencia": 0.10, "saque": 0.05, "compra_negada": 0.04, "troca_senha": 0.01,
}
PROB_PAIS_NORMAL  = {"BR": 0.92, "US": 0.04, "AR": 0.02, "JP": 0.01, "DE": 0.01, "MX": 0.00}
PROB_DEV_NORMAL   = {"celular_habitual": 0.70, "desktop_habitual": 0.25, "celular_novo": 0.04, "dispositivo_desconhecido": 0.01}

# Fase crítica (somente clientes com fraude — últimos 2-4 eventos)
PROB_TIPO_FRAUDE  = {
    "troca_senha": 0.30, "compra_negada": 0.25, "transferencia": 0.20,
    "compra": 0.15, "saque": 0.10, "login": 0.00, "consulta_saldo": 0.00,
}
PROB_PAIS_FRAUDE  = {"BR": 0.05, "US": 0.20, "AR": 0.10, "JP": 0.35, "DE": 0.20, "MX": 0.10}
PROB_DEV_FRAUDE   = {"celular_habitual": 0.05, "desktop_habitual": 0.05, "celular_novo": 0.20, "dispositivo_desconhecido": 0.70}


def amostrar(prob_dict):
    keys = list(prob_dict.keys())
    probs = list(prob_dict.values())
    return random.choices(keys, weights=probs, k=1)[0]


def gerar_cliente(cliente_id, tem_fraude):
    n_eventos = random.randint(MIN_EVENTOS, MAX_EVENTOS)
    eventos = []

    if tem_fraude:
        n_fase_critica = random.randint(2, 4)
        n_normal = n_eventos - n_fase_critica
    else:
        n_normal = n_eventos
        n_fase_critica = 0

    for i in range(n_normal):
        eventos.append({
            "cliente_id":   cliente_id,
            "evento_num":   i,
            "tipo":         amostrar(PROB_TIPO_NORMAL),
            "pais":         amostrar(PROB_PAIS_NORMAL),
            "device":       amostrar(PROB_DEV_NORMAL),
            "valor":        round(random.uniform(0, 500), 2) if random.random() > 0.3 else 0.0,
            "hora":         random.randint(7, 22),  # horário comercial
            "fase_critica": 0,
            "fraude":       int(tem_fraude),
        })

    for j in range(n_fase_critica):
        eventos.append({
            "cliente_id":   cliente_id,
            "evento_num":   n_normal + j,
            "tipo":         amostrar(PROB_TIPO_FRAUDE),
            "pais":         amostrar(PROB_PAIS_FRAUDE),
            "device":       amostrar(PROB_DEV_FRAUDE),
            "valor":        round(random.uniform(500, 10000), 2) if random.random() > 0.4 else 0.0,
            "hora":         random.randint(0, 5),   # madrugada
            "fase_critica": 1,
            "fraude":       int(tem_fraude),
        })

    return eventos


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = time.time()
    bloco(1, "Gerando event stream sintético")

    ids_fraude = set(random.sample(range(N_CLIENTES), N_FRAUD))
    todos_eventos = []
    for cid in range(N_CLIENTES):
        todos_eventos.extend(gerar_cliente(cid, cid in ids_fraude))

    df = pd.DataFrame(todos_eventos)
    df.to_csv(DATA_PATH, index=False)
    ok(f"Arquivo salvo: {DATA_PATH}", t)

    print()
    print(f"  Total de eventos:    {len(df):,}")
    print(f"  Total de clientes:   {N_CLIENTES}")
    n_fraude = df.drop_duplicates('cliente_id')['fraude'].sum()
    print(f"  Clientes com fraude: {n_fraude} ({n_fraude/N_CLIENTES*100:.0f}%)")
    print(f"  Eventos por cliente: {len(df)/N_CLIENTES:.1f} em média")

    bloco(2, "Distribuição de tipos de evento")
    dist = df['tipo'].value_counts()
    for tipo, cnt in dist.items():
        print(f"  {tipo:<25} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

    bloco(3, "Distribuição de países")
    dist_pais = df['pais'].value_counts()
    for pais, cnt in dist_pais.items():
        print(f"  {pais:<6} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")

    bloco(4, "Primeiras linhas")
    print(df.head(10).to_string(index=False))
