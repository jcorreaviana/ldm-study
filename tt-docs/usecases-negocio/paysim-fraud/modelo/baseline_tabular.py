"""
Baseline tabular (XGBoost ou regressao logistica) nas features JA
engenheiradas em preprocessing/preprocess_pipeline.py - objetivo e confirmar
se ha sinal discriminativo alcancavel nos dados antes de continuar ajustando
o transformer sequencial.

Contexto (diagnostico desta sessao): o transformer_sequencial.py, ancorado
em nameDest, nao tem acesso a oldbalanceOrg/newbalanceOrig (saldo de quem
ENVIA) - e e exatamente ai que mora o sinal classico de fraude do PaySim
(TRANSFER que drena ~100% do saldo do remetente, seguido de CASH_OUT
imediato). erroBalanceOrig (oldbalanceOrg - amount - newbalanceOrig) capta
isso diretamente numa unica feature. Este baseline usa o pipeline tabular
completo (que TEM essas colunas) pra checar se esse sinal sozinho ja
discrimina bem - separando "falta sinal no pipeline do transformer" de
"problema de arquitetura/otimizacao do transformer".

Preferencia: XGBoost (lida melhor com nao-linearidade e desbalanceamento via
scale_pos_weight). Se nao estiver instalado (`pip install xgboost`), cai
para LogisticRegression (scikit-learn) com class_weight balanceado.

Nao salva nenhum dataset processado em disco - so le o dataset.csv, treina
em memoria, e salva o relatorio em avaliacao/resultado_baseline.md.

Rode: python baseline_tabular.py --path ../dataset.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent / "preprocessing"))
from preprocess_pipeline import (
    TIPOS,
    calcular_features_velocidade,
    carregar,
    engenharia_features,
    split_por_conta,
)

COLUNAS_NUMERICAS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "erroBalanceOrig", "erroBalanceDest",
    "n_tx_anteriores", "soma_valor_anterior", "max_valor_anterior",
    "soma_valor_janela_curta", "n_tx_janela_curta",
]
COLUNA_BINARIA = "sinal_estruturacao"


def montar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monta a matriz de features: colunas numericas + sinal_estruturacao +
    one-hot de `type`. `type` e categorical com os 5 tipos originais (mesmo
    apos filtrar so TRANSFER/CASH_OUT), entao pd.get_dummies ja cria as 5
    colunas - o loop abaixo so garante isso caso o dtype tenha perdido as
    categorias nao usadas em algum ponto do pipeline.
    """
    dummies_tipo = pd.get_dummies(df["type"], prefix="tipo").astype(int)
    for t in TIPOS:
        col = f"tipo_{t}"
        if col not in dummies_tipo.columns:
            dummies_tipo[col] = 0
    dummies_tipo = dummies_tipo[[f"tipo_{t}" for t in TIPOS]]

    X = pd.concat(
        [
            df[COLUNAS_NUMERICAS].reset_index(drop=True),
            df[[COLUNA_BINARIA]].reset_index(drop=True),
            dummies_tipo.reset_index(drop=True),
        ],
        axis=1,
    )
    return X


def treinar_xgboost(X_treino, y_treino, X_val, y_val):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError(
            "xgboost nao instalado - rode `pip install xgboost` ou use "
            "--modelo logreg"
        ) from e

    scale_pos_weight = (y_treino == 0).sum() / max((y_treino == 1).sum(), 1)
    print(f"XGBoost - scale_pos_weight: {scale_pos_weight:.2f}")

    modelo = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
    )
    modelo.fit(X_treino, y_treino, eval_set=[(X_val, y_val)], verbose=False)
    probs_val = modelo.predict_proba(X_val)[:, 1]
    return modelo, probs_val, "XGBoost"


def treinar_logreg(X_treino, y_treino, X_val, y_val):
    scaler = StandardScaler()
    X_treino_s = scaler.fit_transform(X_treino)
    X_val_s = scaler.transform(X_val)

    modelo = LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1)
    modelo.fit(X_treino_s, y_treino)
    probs_val = modelo.predict_proba(X_val_s)[:, 1]
    return modelo, probs_val, "LogisticRegression"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="caminho local do dataset.csv")
    parser.add_argument("--modelo", choices=["auto", "xgboost", "logreg"], default="auto",
                         help="auto tenta xgboost, cai pra logreg se nao instalado")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = carregar(args.path)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]  # unico lugar com fraude rotulada
    df = engenharia_features(df)
    df = calcular_features_velocidade(df)

    treino_df, val_df, teste_df = split_por_conta(df)
    print(f"linhas treino/val/teste: {len(treino_df):,} / {len(val_df):,} / {len(teste_df):,}")
    print(f"positivas treino/val/teste: {treino_df['isFraud'].sum():,} / "
          f"{val_df['isFraud'].sum():,} / {teste_df['isFraud'].sum():,}")

    X_treino = montar_features(treino_df)
    X_val = montar_features(val_df)
    y_treino = treino_df["isFraud"].to_numpy()
    y_val = val_df["isFraud"].to_numpy()

    nome_modelo_arg = args.modelo
    if nome_modelo_arg == "auto":
        try:
            import xgboost  # noqa: F401
            nome_modelo_arg = "xgboost"
        except ImportError:
            print("[aviso] xgboost nao instalado (pip install xgboost) - "
                  "usando LogisticRegression")
            nome_modelo_arg = "logreg"

    if nome_modelo_arg == "xgboost":
        modelo, probs_val, nome = treinar_xgboost(X_treino, y_treino, X_val, y_val)
    else:
        modelo, probs_val, nome = treinar_logreg(X_treino, y_treino, X_val, y_val)

    auprc = average_precision_score(y_val, probs_val)
    print(f"\n{nome} - AUPRC validacao: {auprc:.4f}")

    # mesmo diagnostico aplicado no transformer - compara diretamente
    probs_fraude = probs_val[y_val == 1]
    probs_legit = probs_val[y_val == 0]
    print(f"scores val fraude   (n={len(probs_fraude)}): media={probs_fraude.mean():.4f} "
          f"mediana={np.median(probs_fraude):.4f}")
    print(f"scores val legitima (n={len(probs_legit)}): media={probs_legit.mean():.4f} "
          f"mediana={np.median(probs_legit):.4f}")
    diferenca = probs_fraude.mean() - probs_legit.mean()
    print(f"diferenca de medias (fraude - legitima): {diferenca:+.4f}")

    importancias = None
    if nome == "XGBoost":
        importancias = sorted(
            zip(X_treino.columns, modelo.feature_importances_), key=lambda x: -x[1]
        )
    elif hasattr(modelo, "coef_"):
        importancias = sorted(
            zip(X_treino.columns, np.abs(modelo.coef_[0])), key=lambda x: -x[1]
        )

    linhas_importancia = "| (sem coeficientes/importancia disponivel) | - |"
    if importancias:
        print("\ntop 10 features mais importantes:")
        for col, imp in importancias[:10]:
            print(f"  {col:24s} {imp:.4f}")
        linhas_importancia = "\n".join(
            f"| {col} | {imp:.4f} |" for col, imp in importancias[:10]
        )

    preds_bin = (probs_val >= args.threshold).astype(int)
    recall = recall_score(y_val, preds_bin)
    cm = confusion_matrix(y_val, preds_bin)
    tn, fp, fn, tp = cm.ravel()

    resultado = f"""# Resultado do baseline tabular ({nome})

Contexto: o transformer sequencial (ancorado em `nameDest`) nao tem acesso a
`oldbalanceOrg`/`newbalanceOrig` (saldo de quem envia) - onde mora o sinal
classico de fraude do PaySim (TRANSFER que drena ~100% do saldo do
remetente, seguido de CASH_OUT imediato). Este baseline usa as features
tabulares completas (incluindo `erroBalanceOrig`) para checar se ha sinal
discriminativo alcancavel, antes de continuar ajustando o transformer.

## Dados

| | Treino | Validação | Teste |
|---|---|---|---|
| Linhas | {len(treino_df):,} | {len(val_df):,} | {len(teste_df):,} |
| Positivas (fraude) | {int(treino_df['isFraud'].sum()):,} | {int(val_df['isFraud'].sum()):,} | {int(teste_df['isFraud'].sum()):,} |

Split por `nameDest` (mesma função `split_por_conta`, seed=42, usada em
`preprocess_pipeline.py` e `montar_event_stream.py`) — nenhuma conta se
repete entre treino e validação.

## Resultado

| Métrica | Valor |
|---|---|
| Modelo | {nome} |
| AUPRC (validação) | {auprc:.4f} |
| Recall (threshold={args.threshold}) | {recall:.4f} |
| TP / FP / FN / TN | {tp} / {fp} / {fn} / {tn} |
| Score médio — fraude | {probs_fraude.mean():.4f} |
| Score médio — legítima | {probs_legit.mean():.4f} |
| Diferença de médias | {diferenca:+.4f} |

Para comparação: o transformer sequencial tinha diferença de médias de
0.0001 (não discriminava nada). Se este baseline mostrar diferença
substancialmente maior, confirma que o problema do transformer é falta de
acesso às features do remetente (`oldbalanceOrg`/`newbalanceOrig`/
`erroBalanceOrig`), não arquitetura/otimização.

## Features mais importantes

| Feature | Importância |
|---|---|
{linhas_importancia}

## Comando para reproduzir

```
python baseline_tabular.py --path ../dataset.csv --modelo {nome_modelo_arg}
```
"""
    out_path = Path(__file__).parent.parent / "avaliacao" / "resultado_baseline.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resultado)
    print(f"\nresultado salvo em {out_path}")


if __name__ == "__main__":
    main()
