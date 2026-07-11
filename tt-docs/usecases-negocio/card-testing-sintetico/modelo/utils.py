"""
Funções auxiliares do transformer sequencial: carga de dados, split por
clienteID (nunca por transação — isso vazaria contexto de sequência entre
treino/val/teste), pré-processamento (padronização numérica + vocabulário
categórico) e identificação das micro-transações precursoras do golpe
(usadas na métrica de negócio de evaluate.py).

Layout de pastas esperado:
    card-testing-sintetico/
        dataset_sintetico.csv
        eda/            <- EDA (já entregue)
        baseline/        <- baseline XGBoost (já entregue)
        modelo/
            config.py
            utils.py             <- este arquivo
            dataset.py
            model.py
            train.py
            evaluate.py
            modelo_out/          <- checkpoint, preprocessadores, métricas
"""

import pickle
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config

FEATURES_NUMERICAS = ["valor", "hora", "saldo_antes", "saldo_depois"]
FEATURES_CATEGORICAS = ["tipo", "merchant"]
TARGET = "isFraud"

PAD_IDX = 0  # índice reservado para padding nas embeddings categóricas


def load_data(cfg: Config) -> pd.DataFrame:
    """Carrega o dataset, remove linhas vazias e cria a coluna 'hora'."""
    df = pd.read_csv(cfg.data_path, parse_dates=["timestamp"])
    df = df.dropna(subset=["clienteID"]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df["hora"] = df["timestamp"].dt.hour
    df = df.sort_values(["clienteID", "timestamp"]).reset_index(drop=True)
    return df


def split_clientes(df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split estratificado por clienteID (não por transação), preservando a
    proporção de clientes fraudadores em cada parte. A sequência inteira de
    um cliente sempre fica em um único split.
    """
    clientes = df["clienteID"].unique()
    tem_fraude = df.groupby("clienteID")[TARGET].max().reindex(clientes).values

    train_ids, temp_ids, train_y, temp_y = train_test_split(
        clientes, tem_fraude,
        test_size=cfg.val_size + cfg.test_size,
        stratify=tem_fraude,
        random_state=cfg.seed,
    )
    val_frac_of_temp = cfg.val_size / (cfg.val_size + cfg.test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1 - val_frac_of_temp,
        stratify=temp_y,
        random_state=cfg.seed,
    )
    return train_ids, val_ids, test_ids


@dataclass
class Preprocessors:
    numeric_mean: pd.Series
    numeric_std: pd.Series
    tipo2idx: Dict[str, int]
    merchant2idx: Dict[str, int]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "Preprocessors":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_preprocessors(df_train: pd.DataFrame) -> Preprocessors:
    """
    Ajusta padronização (z-score) das numéricas e vocabulário das
    categóricas usando SOMENTE as transações de treino (evita vazamento).
    Índice 0 é reservado para padding; categorias começam em 1.
    """
    numeric_mean = df_train[FEATURES_NUMERICAS].mean()
    numeric_std = df_train[FEATURES_NUMERICAS].std().replace(0, 1.0)

    tipo2idx = {v: i + 1 for i, v in enumerate(sorted(df_train["tipo"].unique()))}
    merchant2idx = {v: i + 1 for i, v in enumerate(sorted(df_train["merchant"].unique()))}

    return Preprocessors(numeric_mean, numeric_std, tipo2idx, merchant2idx)


def transform(df: pd.DataFrame, prep: Preprocessors) -> pd.DataFrame:
    """
    Aplica a padronização/vocabulário já ajustados. Categorias não vistas no
    treino (não deveria acontecer com este dataset, mas por robustez) caem
    no índice PAD_IDX, o que efetivamente as trata como token neutro.
    """
    df = df.copy()
    for col in FEATURES_NUMERICAS:
        df[f"{col}_norm"] = (df[col] - prep.numeric_mean[col]) / prep.numeric_std[col]

    df["tipo_idx"] = df["tipo"].map(prep.tipo2idx).fillna(PAD_IDX).astype(int)
    df["merchant_idx"] = df["merchant"].map(prep.merchant2idx).fillna(PAD_IDX).astype(int)
    return df


def identify_micro_precursors(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Marca as transações que são "micro-transações precursoras de golpe":
    as `cfg.n_precursores` transações imediatamente anteriores a uma fraude
    confirmada (isFraud=1), desde que todas tenham valor < cfg.micro_threshold.

    Essas transações têm isFraud=0 (rótulo real) mas representam exatamente
    o padrão de card testing que o negócio quer pegar ANTES do golpe. Servem
    de GABARITO para a métrica de negócio em evaluate.py (que compara a
    probabilidade prevista contra esse gabarito, não contra isFraud).

    Diferente de propagate_label_preventivo (usada para o LABEL DE TREINO),
    esta função exige que as transações anteriores tenham valor abaixo de
    micro_threshold -- é uma identificação mais estrita, pensada para medir
    "o modelo pegou especificamente o padrão de micro-transação", enquanto
    o label de treino é propagado só pela posição (ver função abaixo).

    Retorna o df original com uma coluna booleana extra: 'is_micro_precursor'.
    """
    df = df.sort_values(["clienteID", "timestamp"]).reset_index(drop=True)
    df["is_micro_precursor"] = False

    fraud_clients = df.loc[df[TARGET] == 1, "clienteID"].unique()
    for cid in fraud_clients:
        idx_cliente = df.index[df["clienteID"] == cid]
        grp = df.loc[idx_cliente]
        posicoes_golpe = np.where(grp[TARGET].values == 1)[0]
        for pos in posicoes_golpe:
            if pos >= cfg.n_precursores:
                janela = grp.iloc[pos - cfg.n_precursores:pos]
                if (janela["valor"] < cfg.micro_threshold).all():
                    df.loc[janela.index, "is_micro_precursor"] = True

    return df


def propagate_label_preventivo(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Cria a coluna 'label_preventivo' -- o NOVO alvo de treino, propagando o
    sinal de fraude para trás no tempo dentro da sequência do mesmo cliente.

    Motivação (achado do AUPRC=1.0 com recall nas micro-transações=0%): ao
    treinar contra isFraud puro, o modelo aprende — corretamente, de acordo
    com o rótulo — que as micro-transações NÃO são fraude, porque de fato
    isFraud=0 nelas. Ele fica ótimo em achar o golpe (que tem sinal isolado
    forte: valor alto, madrugada) e nunca aprende a associar o padrão
    micro -> micro -> golpe, porque nunca recebeu gradiente nesse sentido.

    Regra: para cada transação de golpe confirmado (isFraud=1) de um cliente,
    marca ELA MESMA e as `cfg.n_precursores` transações imediatamente
    anteriores do mesmo cliente com label_preventivo=1 -- mesmo que o isFraud
    original dessas anteriores seja 0. Todas as demais transações ficam com
    label_preventivo=0. Ao contrário de identify_micro_precursors, aqui NÃO
    se exige valor < micro_threshold: a propagação é só por posição relativa
    ao golpe, porque queremos que o modelo generalize o padrão de sequência,
    não decore um corte de valor específico.

    NÃO modifica a coluna 'isFraud' original -- ela continua disponível
    intacta para quem precisar do rótulo real (ex.: métricas de avaliação).
    Esta função só adiciona a coluna nova 'label_preventivo'.
    """
    df = df.sort_values(["clienteID", "timestamp"]).reset_index(drop=True)
    # começa igual ao isFraud (o golpe em si já é label=1) e depois propaga
    # para trás as transações precursoras
    df["label_preventivo"] = df[TARGET].astype(int)

    fraud_clients = df.loc[df[TARGET] == 1, "clienteID"].unique()
    for cid in fraud_clients:
        idx_cliente = df.index[df["clienteID"] == cid]
        grp = df.loc[idx_cliente]
        posicoes_golpe = np.where(grp[TARGET].values == 1)[0]
        for pos in posicoes_golpe:
            inicio = max(0, pos - cfg.n_precursores)
            indices_anteriores = grp.iloc[inicio:pos].index
            df.loc[indices_anteriores, "label_preventivo"] = 1

    return df
