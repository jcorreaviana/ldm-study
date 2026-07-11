"""
Funções auxiliares compartilhadas pelos scripts de EDA do dataset de card testing.

Layout de pastas esperado:
    card-testing-sintetico/
        dataset_sintetico.csv
        eda/
            utils.py          <- este arquivo
            01_shape_dtypes.py
            02_distribuicao_fraude.py
            03_distribuicao_tipos.py
            04_sequencia_fraudador.py
            05_sequencia_legitimo.py
            06_distribuicao_valores.py
            07_distribuicao_horarios.py
            eda_out/           <- gráficos gerados (criado automaticamente)
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "dataset_sintetico.csv"
OUT_DIR = HERE / "eda_out"
OUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    """Carrega o dataset sintético e normaliza tipos/colunas derivadas."""
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values(["clienteID", "timestamp"]).reset_index(drop=True)
    df["hora"] = df["timestamp"].dt.hour
    return df


def find_fraud_example(df: pd.DataFrame, micro_threshold: float = 10.0) -> str:
    """
    Procura um clienteID fraudador cuja transação de fraude foi precedida
    por pelo menos duas micro-transações (valor < micro_threshold).
    Isso é o padrão clássico de card testing: micro -> micro -> golpe.
    Faz fallback para o primeiro cliente fraudador caso nenhum siga o padrão.
    """
    fraud_clients = df.loc[df["isFraud"] == 1, "clienteID"].unique()

    for cid in fraud_clients:
        grp = df[df["clienteID"] == cid].sort_values("timestamp").reset_index(drop=True)
        fraud_positions = grp.index[grp["isFraud"] == 1].tolist()
        for pos in fraud_positions:
            if pos >= 2:
                anteriores = grp.iloc[pos - 2:pos]
                if (anteriores["valor"] < micro_threshold).all():
                    return cid
    return fraud_clients[0]


def find_legit_example(df: pd.DataFrame, min_transacoes: int = 5) -> str:
    """
    Procura um clienteID sem nenhuma fraude e com um número razoável de
    transações, para servir de contraponto ao cliente fraudador.
    """
    fraud_clients = set(df.loc[df["isFraud"] == 1, "clienteID"].unique())
    contagens = df[~df["clienteID"].isin(fraud_clients)].groupby("clienteID").size()
    candidatos = contagens[contagens >= min_transacoes].index
    return candidatos[0]
