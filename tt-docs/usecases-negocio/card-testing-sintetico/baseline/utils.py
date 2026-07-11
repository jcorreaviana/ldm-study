"""
Funções auxiliares compartilhadas pelos scripts do baseline XGBoost.

Layout de pastas esperado:
    card-testing-sintetico/
        dataset_sintetico.csv
        eda/                       <- EDA (já entregue)
        baseline/
            utils.py                <- este arquivo
            01_train_baseline.py
            02_avaliar_baseline.py
            baseline_out/           <- métricas e gráficos gerados

IMPORTANTE — escopo deste baseline:
Este modelo usa APENAS features de transação isolada (valor, tipo, merchant,
saldo_antes, saldo_depois, hora do dia). Nenhuma feature de sequência/histórico
do cliente é usada de propósito: o objetivo é reproduzir o comportamento do
modelo atual (transação a transação) e medir, com números, a limitação que a
EDA já mostrou visualmente — a saber, que as micro-transações de teste de
cartão são indistinguíveis de qualquer outra compra pequena legítima quando
olhadas isoladamente.
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "dataset_sintetico.csv"
OUT_DIR = HERE / "baseline_out"
OUT_DIR.mkdir(exist_ok=True)

FEATURES_NUMERICAS = ["valor", "saldo_antes", "saldo_depois", "hora"]
FEATURES_CATEGORICAS = ["tipo", "merchant"]
TARGET = "isFraud"


def load_data() -> pd.DataFrame:
    """Carrega o dataset, remove linhas totalmente vazias e cria a coluna 'hora'."""
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.dropna(subset=["clienteID"]).copy()  # remove eventual linha em branco no final do CSV
    df[TARGET] = df[TARGET].astype(int)
    df["hora"] = df["timestamp"].dt.hour
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame):
    """
    Constrói a matriz de features X (apenas transação isolada) e o target y.
    Categóricas viram one-hot. Retorna X, y e a lista de clienteID/timestamp
    (mantidos à parte, não entram no modelo, só para rastreabilidade na avaliação).
    """
    X_num = df[FEATURES_NUMERICAS].copy()
    X_cat = pd.get_dummies(df[FEATURES_CATEGORICAS], prefix=FEATURES_CATEGORICAS)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df[TARGET].copy()
    meta = df[["clienteID", "timestamp"]].copy()
    return X, y, meta
