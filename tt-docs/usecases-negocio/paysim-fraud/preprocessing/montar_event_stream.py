"""
Monta o event stream por conta de destino (nameDest) - ver EDA_paysim.md
secao 7 para a justificativa de por que nameDest e nao nameOrig/clienteID.

Para cada nameDest:
  - ordena as transacoes recebidas por `step` crescente
  - corta a sequencia em pontos de decisao SEM VAZAMENTO: ao avaliar o evento
    na posicao k, o contexto usado sao so os eventos 0..k-1 (nunca o proprio
    evento rotulado, nem eventos futuros)
  - trunca/faz padding para um comprimento fixo de janela (--janela)

Saida: arrays numpy (.npz) com tensores prontos para o Dataset do PyTorch.
NAO salva CSV nem duplica o dataset bruto - so os tensores derivados, e num
diretorio fora do controle de versao do projeto (--output-dir).

Rode: python montar_event_stream.py --path ../dataset.csv --output-dir ../../paysim_data --janela 13

Janela=13 confirmada por analisar_namedest.py (censo completo, nao amostra):
p90 do numero de transacoes por nameDest = 13, cobre 90% das contas sem
truncar. Mediana real é so 3 (curta demais para o transformer aprender
padrao sequencial) e VRAM da RTX 5060 (~8.5GB) limita janelas maiores.
Ver EDA_paysim.md secao 7 e avaliacao/resultado_final.md para o registro
completo da decisao.

CORRECAO (diagnostico confirmado: logit nao-finito ja no forward pass, antes
da loss): amount/oldbalanceDest/newbalanceDest chegam a dezenas de milhoes
sem normalizacao (ver EDA_paysim.md) e alimentavam o nn.Linear direto,
causando overflow. Agora:
  - split por conta (mesma funcao split_por_conta de preprocess_pipeline.py,
    mesma seed) ANTES de montar as sequencias
  - z-score de amount, delta_step, oldbalanceDest, newbalanceDest calculado
    SO nas contas de treino, aplicado nos tres splits (mesmo padrao usado no
    projeto de credit card)
  - as estatisticas vao para preprocessing_meta.json (chave "event_stream")
    para uso na inferencia em producao
  - o array "split" (treino/val/teste) e salvo junto no .npz, por conta -
    transformer_sequencial.py ainda faz o proprio random_split ignorando
    esse campo (nao alterado agora, fora do escopo pedido); ajustar la para
    consumir esse campo e evitar vazamento de conta entre treino/val.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from preprocess_pipeline import atualizar_meta_json, split_por_conta

TIPOS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TIPO_TO_IDX = {t: i for i, t in enumerate(TIPOS)}


def calcular_delta_step(df: pd.DataFrame) -> pd.Series:
    """delta_step por nameDest (steps desde a transacao anterior da mesma
    conta) - calculado so para poder tirar media/std do split de treino."""
    df_ordenado = df.sort_values(["nameDest", "step"])
    return df_ordenado.groupby("nameDest")["step"].diff().fillna(0)


def _stats(serie: pd.Series) -> dict:
    return {"media": float(serie.mean()), "std": float(serie.std() or 1.0)}


def calcular_zscore_treino(treino_df: pd.DataFrame) -> dict:
    """
    Z-score calculado SO nas contas do split de treino - nunca usar val/teste
    aqui, senao vaza informacao do split de avaliacao para a normalizacao.
    """
    delta_step_treino = calcular_delta_step(treino_df)
    return {
        "amount": _stats(treino_df["amount"]),
        "oldbalanceDest": _stats(treino_df["oldbalanceDest"]),
        "newbalanceDest": _stats(treino_df["newbalanceDest"]),
        "delta_step": _stats(delta_step_treino),
    }


def montar_sequencias(df: pd.DataFrame, janela: int, stats_zscore: dict):
    """
    Retorna, para cada (nameDest, posicao k) com k >= 1:
      - contexto: ate `janela` eventos anteriores aquela conta (posicoes 0..k-1),
        com amount/delta_step/oldbalanceDest/newbalanceDest ja normalizados
        (z-score de stats_zscore, calculado so no treino)
      - label: isFraud do evento na posicao k (o evento sendo avaliado agora,
        NAO incluido no contexto)

    Posicoes de padding continuam 0.0 literal (o np.zeros inicial nunca e
    sobrescrito nelas) - nao normalizamos o padding, e o
    src_key_padding_mask do transformer ja exclui essas posicoes da atencao,
    entao valores "estranhos" ali nao afetam o treino.
    """
    df = df.sort_values(["nameDest", "step"]).reset_index(drop=True)

    m_amount, s_amount = stats_zscore["amount"]["media"], stats_zscore["amount"]["std"]
    m_delta, s_delta = stats_zscore["delta_step"]["media"], stats_zscore["delta_step"]["std"]
    m_old, s_old = stats_zscore["oldbalanceDest"]["media"], stats_zscore["oldbalanceDest"]["std"]
    m_new, s_new = stats_zscore["newbalanceDest"]["media"], stats_zscore["newbalanceDest"]["std"]

    feats = []
    labels = []
    seq_lens = []
    dest_ids = []

    for name_dest, grupo in df.groupby("nameDest", sort=False):
        grupo = grupo.reset_index(drop=True)
        n = len(grupo)
        if n < 2:
            continue  # sem contexto anterior, nao da para avaliar sem vazamento

        tipo_idx = grupo["type"].map(TIPO_TO_IDX).to_numpy()
        amount = ((grupo["amount"].to_numpy(dtype="float64") - m_amount) / s_amount).astype("float32")
        delta_step_raw = grupo["step"].diff().fillna(0).to_numpy(dtype="float64")
        delta_step = ((delta_step_raw - m_delta) / s_delta).astype("float32")
        old_bal = ((grupo["oldbalanceDest"].to_numpy(dtype="float64") - m_old) / s_old).astype("float32")
        new_bal = ((grupo["newbalanceDest"].to_numpy(dtype="float64") - m_new) / s_new).astype("float32")
        fraud = grupo["isFraud"].to_numpy(dtype="int8")

        for k in range(1, n):
            ini = max(0, k - janela)
            ctx_len = k - ini
            ctx = np.zeros((janela, 5), dtype="float32")
            ctx[-ctx_len:, 0] = tipo_idx[ini:k]
            ctx[-ctx_len:, 1] = amount[ini:k]
            ctx[-ctx_len:, 2] = delta_step[ini:k]
            ctx[-ctx_len:, 3] = old_bal[ini:k]
            ctx[-ctx_len:, 4] = new_bal[ini:k]

            feats.append(ctx)
            labels.append(fraud[k])
            seq_lens.append(ctx_len)
            dest_ids.append(name_dest)

    if not feats:
        return (
            np.zeros((0, janela, 5), dtype="float32"),
            np.zeros((0,), dtype="int8"),
            np.zeros((0,), dtype="int32"),
            np.zeros((0,), dtype=object),
        )

    return (
        np.stack(feats),
        np.array(labels, dtype="int8"),
        np.array(seq_lens, dtype="int32"),
        np.array(dest_ids, dtype=object),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--output-dir", default="../../paysim_data")
    parser.add_argument("--janela", type=int, default=13,
                         help="comprimento da janela de contexto - 13 confirmado "
                              "por analisar_namedest.py (p90), ver EDA_paysim.md 7.2")
    parser.add_argument("--seed", type=int, default=42,
                         help="mesma seed do split_por_conta de preprocess_pipeline.py - "
                              "PRECISA ser igual nos dois scripts, senao uma conta pode "
                              "cair em treino num script e em teste no outro")
    args = parser.parse_args()

    dtypes = {
        "step": "int32", "type": "category", "amount": "float64",
        "nameDest": "string", "oldbalanceDest": "float64",
        "newbalanceDest": "float64", "isFraud": "int8",
    }
    df = pd.read_csv(args.path, dtype=dtypes, usecols=list(dtypes.keys()))
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]

    treino_df, val_df, teste_df = split_por_conta(df, seed=args.seed)
    print(f"contas treino/val/teste: {treino_df['nameDest'].nunique():,} / "
          f"{val_df['nameDest'].nunique():,} / {teste_df['nameDest'].nunique():,}")

    stats_zscore = calcular_zscore_treino(treino_df)
    print("z-score calculado SO no split de treino:")
    for coluna, s in stats_zscore.items():
        print(f"  {coluna:16s} media={s['media']:>14.4f}  std={s['std']:>14.4f}")

    X_treino, y_treino, sl_treino, id_treino = montar_sequencias(treino_df, args.janela, stats_zscore)
    X_val, y_val, sl_val, id_val = montar_sequencias(val_df, args.janela, stats_zscore)
    X_teste, y_teste, sl_teste, id_teste = montar_sequencias(teste_df, args.janela, stats_zscore)

    X = np.concatenate([X_treino, X_val, X_teste])
    y = np.concatenate([y_treino, y_val, y_teste])
    seq_lens = np.concatenate([sl_treino, sl_val, sl_teste])
    dest_ids = np.concatenate([id_treino, id_val, id_teste])
    split = np.array(
        ["treino"] * len(y_treino) + ["val"] * len(y_val) + ["teste"] * len(y_teste),
        dtype=object,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "event_stream.npz",
        X=X, y=y, seq_lens=seq_lens, dest_ids=dest_ids, split=split,
    )
    print(f"salvo em {out / 'event_stream.npz'}")
    print(f"amostras treino/val/teste: {len(y_treino):,} / {len(y_val):,} / {len(y_teste):,}")
    print(f"positivas treino/val/teste: {int(y_treino.sum()):,} / "
          f"{int(y_val.sum()):,} / {int(y_teste.sum()):,}")

    meta_novas_chaves = {
        "event_stream": {
            "janela": args.janela,
            "tipos": TIPOS,
            "entidade_sequencia": "nameDest",
            "corte_sem_vazamento": True,
            "split_seed": args.seed,
            "scaler_zscore": stats_zscore,
            "n_treino": int(len(y_treino)),
            "n_val": int(len(y_val)),
            "n_teste": int(len(y_teste)),
            "n_positivas_treino": int(y_treino.sum()),
            "n_positivas_val": int(y_val.sum()),
            "n_positivas_teste": int(y_teste.sum()),
        }
    }
    meta_path = atualizar_meta_json(meta_novas_chaves)
    print(f"estatisticas de normalizacao salvas em {meta_path} (chave 'event_stream')")


if __name__ == "__main__":
    main()
