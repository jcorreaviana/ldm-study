"""
Pipeline de pre-processamento tabular (complementar ao montar_event_stream.py,
que cuida especificamente do agrupamento em sequencias).

Etapas:
  1. carregar e remover duplicatas exatas
  2. engenharia de features de saldo (erroBalanceOrig/Dest - inconsistencias
     entre saldo declarado antes/depois e o valor movimentado; sinal classico
     de fraude no PaySim)
  3. features de velocidade por nameDest (complemento tabular ao event stream
     bruto de montar_event_stream.py - ver docstring de
     calcular_features_velocidade)
  4. calcular estatisticas de normalizacao (z-score) SO no split de treino
  5. split por nameDest (nao por transacao) para nao vazar a mesma conta
     entre treino/val/teste
  6. salvar `preprocessing_meta.json` com o scaler e o vocabulario de tipos

Datasets processados NAO sao salvos neste repositorio (so os scripts e o
meta). Aponte --output-dir para uma pasta fora do controle de versao.

Nota: o label por posicao e o corte sem vazamento na SEQUENCIA (o que o
transformer realmente consome) continuam vivendo em montar_event_stream.py
e nao mudam aqui. As features de velocidade abaixo sao tabulares/agregadas -
uteis como baseline (ex.: gradient boosting) e como sinal adicional, mas nao
substituem o event stream bruto.

Rode: python preprocess_pipeline.py --path ../dataset.csv --output-dir ../../paysim_data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TIPOS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def carregar(path: str) -> pd.DataFrame:
    dtypes = {
        "step": "int32", "type": "category", "amount": "float64",
        "nameOrig": "string", "oldbalanceOrg": "float64",
        "newbalanceOrig": "float64", "nameDest": "string",
        "oldbalanceDest": "float64", "newbalanceDest": "float64",
        "isFraud": "int8", "isFlaggedFraud": "int8",
    }
    df = pd.read_csv(path, dtype=dtypes)
    antes = len(df)
    df = df.drop_duplicates()
    print(f"duplicatas removidas: {antes - len(df):,}")
    return df


def engenharia_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # inconsistencia de saldo - sinal classico de fraude no PaySim
    df["erroBalanceOrig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    )
    df["erroBalanceDest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )
    df["amount_pct_limite"] = df["amount"] / 10_000.0
    return df


def calcular_features_velocidade(
    df: pd.DataFrame, janela_curta: int = 6, limite: float = 10_000.0
) -> pd.DataFrame:
    """
    Features de velocidade por nameDest (conta de destino), a versao tabular
    do que build_velocity_features() calculava antes de decidirmos usar
    nameDest em vez de nameOrig como ancora do event stream.

    Cada feature usa SO transacoes anteriores aquela linha (shift antes de
    qualquer agregacao) - o mesmo corte sem vazamento aplicado em
    montar_event_stream.py, so que aqui em forma de agregados por linha em
    vez de sequencia completa. Isso preserva a compatibilidade com o label
    por posicao: para a transacao k, essas features só enxergam 0..k-1.

    Colunas geradas:
      - n_tx_anteriores: quantas transacoes essa conta ja recebeu antes desta
      - soma_valor_anterior: soma de tudo que essa conta recebeu antes desta
      - max_valor_anterior: maior transacao individual recebida antes desta
      - soma_valor_janela_curta / n_tx_janela_curta: mesmas contas, mas so
        nas ultimas `janela_curta` transacoes anteriores (sinal de rajada)
      - sinal_estruturacao: 1 se a soma recente ja passou do limite de
        bloqueio mas nenhuma transacao individual anterior passou sozinha -
        a assinatura de "card testing" traduzida para conta de destino
        (varias entradas fracionadas que so juntas ultrapassam o limite)
    """
    df = df.sort_values(["nameDest", "step"]).copy()
    grupo_amount = df.groupby("nameDest")["amount"]

    df["n_tx_anteriores"] = df.groupby("nameDest").cumcount()
    df["soma_valor_anterior"] = grupo_amount.cumsum() - df["amount"]
    df["max_valor_anterior"] = grupo_amount.transform(lambda s: s.shift(1).cummax()).fillna(0)

    df["soma_valor_janela_curta"] = grupo_amount.transform(
        lambda s: s.shift(1).rolling(janela_curta, min_periods=1).sum()
    ).fillna(0)
    df["n_tx_janela_curta"] = grupo_amount.transform(
        lambda s: s.shift(1).rolling(janela_curta, min_periods=1).count()
    ).fillna(0)

    df["sinal_estruturacao"] = (
        (df["soma_valor_janela_curta"] >= limite)
        & (df["max_valor_anterior"] < limite)
        & (df["n_tx_anteriores"] > 0)
    ).astype("int8")

    return df


def split_por_conta(df: pd.DataFrame, seed: int = 42):
    """
    Split estratificado por nameDest (NAO por transacao) - garante que a
    mesma conta nao aparece em treino e teste ao mesmo tempo, evitando
    vazamento (ver EDA_paysim.md 7.5).
    """
    contas = df["nameDest"].drop_duplicates()
    rng = np.random.RandomState(seed)
    contas = contas.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(contas)
    n_treino = int(n * 0.7)
    n_val = int(n * 0.15)

    treino_contas = set(contas[:n_treino])
    val_contas = set(contas[n_treino:n_treino + n_val])
    teste_contas = set(contas[n_treino + n_val:])

    return (
        df[df["nameDest"].isin(treino_contas)],
        df[df["nameDest"].isin(val_contas)],
        df[df["nameDest"].isin(teste_contas)],
    )


def calcular_scaler(df_treino: pd.DataFrame, colunas: list[str]) -> dict:
    stats = {}
    for c in colunas:
        stats[c] = {
            "media": float(df_treino[c].mean()),
            "std": float(df_treino[c].std() or 1.0),
        }
    return stats


def atualizar_meta_json(novas_chaves: dict) -> Path:
    """
    Le/mescla/escreve preprocessing_meta.json em vez de sobrescrever o
    arquivo inteiro. Necessario porque tanto este script quanto
    montar_event_stream.py escrevem chaves nesse mesmo arquivo - sem merge,
    o que rodar por ultimo apaga o que o outro tinha escrito.
    """
    meta_path = Path(__file__).parent / "preprocessing_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                meta = {}
    meta.update(novas_chaves)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--output-dir", default="../../paysim_data")
    args = parser.parse_args()

    df = carregar(args.path)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]  # unico lugar com fraude
    df = engenharia_features(df)
    df = calcular_features_velocidade(df)

    n_sinalizadas = df["sinal_estruturacao"].sum()
    print(f"transacoes com sinal_estruturacao=1: {n_sinalizadas:,} "
          f"({n_sinalizadas / len(df) * 100:.4f}%)")
    print(f"dessas, quantas sao fraude de verdade (isFraud=1): "
          f"{df.loc[df['sinal_estruturacao'] == 1, 'isFraud'].sum():,}")

    treino, val, teste = split_por_conta(df)
    print(f"contas treino/val/teste: {treino['nameDest'].nunique():,} / "
          f"{val['nameDest'].nunique():,} / {teste['nameDest'].nunique():,}")
    print(f"linhas treino/val/teste: {len(treino):,} / {len(val):,} / {len(teste):,}")

    colunas_numericas = [
        "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "erroBalanceOrig", "erroBalanceDest",
        "n_tx_anteriores", "soma_valor_anterior", "max_valor_anterior",
        "soma_valor_janela_curta", "n_tx_janela_curta",
    ]
    scaler = calcular_scaler(treino, colunas_numericas)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "tabular": {
            "colunas_numericas": colunas_numericas,
            "colunas_binarias": ["sinal_estruturacao"],
            "scaler_zscore": scaler,
            "vocabulario_tipo": TIPOS,
            "split": "por nameDest, 70/15/15",
            "n_treino": len(treino),
            "n_val": len(val),
            "n_teste": len(teste),
            "n_sinalizadas_estruturacao": int(n_sinalizadas),
            "class_weight_sugerido": {
                "0": 1.0,
                "1": float(len(df) / max(df["isFraud"].sum(), 1) / 2),
            },
        }
    }
    meta_path = atualizar_meta_json(meta)
    print(f"meta salvo em {meta_path} (chave 'tabular' - "
          f"'event_stream' e escrita por montar_event_stream.py, sem sobrescrever esta)")


if __name__ == "__main__":
    main()
