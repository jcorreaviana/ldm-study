"""
1. Treina o baseline XGBoost usando apenas features de transação isolada.

Split: aleatório estratificado (não temporal) em treino / validação / teste,
70% / 15% / 15%, preservando a proporção de fraude em cada parte. Como o
dataset é sintético e as datas não representam uma evolução real, o split
temporal introduziria viés artificial sem benefício — em produção, com dados
reais, o correto seria split temporal.
"""

import json
import pickle

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from utils import load_data, build_features, OUT_DIR

RANDOM_STATE = 42


def main():
    df = load_data()
    X, y, meta = build_features(df)

    print("=" * 60)
    print("FEATURES USADAS (apenas transação isolada)")
    print("=" * 60)
    print(list(X.columns))
    print(f"\nShape de X: {X.shape}")
    print(f"Taxa de fraude geral: {y.mean() * 100:.3f}%")

    # split 70/15/15 estratificado
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    print()
    print("=" * 60)
    print("SPLIT ESTRATIFICADO (70% treino / 15% val / 15% teste)")
    print("=" * 60)
    for nome, y_split in [("treino", y_train), ("validação", y_val), ("teste", y_test)]:
        print(
            f"{nome:10s}: n={len(y_split):>6,} | fraudes={int(y_split.sum()):>4} "
            f"| taxa={y_split.mean() * 100:.3f}%".replace(",", ".")
        )

    # scale_pos_weight compensa o desbalanceamento severo (~0.71% de fraude)
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"\nscale_pos_weight (neg/pos no treino): {scale_pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        early_stopping_rounds=30,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"\nMelhor iteração (early stopping): {model.best_iteration}")

    # salva modelo e splits para o script de avaliação
    with open(OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    X_test.to_parquet(OUT_DIR / "X_test.parquet")
    y_test.to_frame().to_parquet(OUT_DIR / "y_test.parquet")
    meta.loc[X_test.index].to_parquet(OUT_DIR / "meta_test.parquet")

    with open(OUT_DIR / "feature_names.json", "w") as f:
        json.dump(list(X.columns), f)

    with open(OUT_DIR / "split_info.json", "w") as f:
        json.dump(
            {
                "n_train": int(len(y_train)),
                "n_val": int(len(y_val)),
                "n_test": int(len(y_test)),
                "fraude_train_pct": float(y_train.mean() * 100),
                "fraude_val_pct": float(y_val.mean() * 100),
                "fraude_test_pct": float(y_test.mean() * 100),
                "scale_pos_weight": float(scale_pos_weight),
                "best_iteration": int(model.best_iteration) if model.best_iteration is not None else None,
            },
            f,
            indent=2,
        )

    print(f"\nModelo e artefatos salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
