import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Dados ──────────────────────────────────────────────────────────────────────
N_CLIENTES  = 200
N_FRAUD     = 60      # 30% de fraude
MAX_EVENTOS = 12
MIN_EVENTOS = 5

# ── Modelo ─────────────────────────────────────────────────────────────────────
SEQ_LEN  = 10   # padding/truncate para essa quantidade de eventos
D_MODEL  = 32   # dimensão dos embeddings
N_HEADS  = 4    # cabeças de atenção
N_LAYERS = 2    # camadas do transformer
D_FF     = 64   # feed-forward interno
DROPOUT  = 0.1

# ── Treino ─────────────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 32
LR         = 0.001

# ── Vocabulários (fixos — não aprendidos) ──────────────────────────────────────
TIPOS = {
    "login":           0,
    "compra":          1,
    "compra_negada":   2,
    "troca_senha":     3,
    "consulta_saldo":  4,
    "transferencia":   5,
    "saque":           6,
}

PAISES = {
    "BR": 0,
    "US": 1,
    "AR": 2,
    "JP": 3,
    "DE": 4,
    "MX": 5,
}

DEVICES = {
    "celular_habitual":        0,
    "celular_novo":            1,
    "desktop_habitual":        2,
    "dispositivo_desconhecido": 3,
}

# ── Reversos (índice → nome) ───────────────────────────────────────────────────
TIPOS_INV   = {v: k for k, v in TIPOS.items()}
PAISES_INV  = {v: k for k, v in PAISES.items()}
DEVICES_INV = {v: k for k, v in DEVICES.items()}

# ── Paths ──────────────────────────────────────────────────────────────────────
import os
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
