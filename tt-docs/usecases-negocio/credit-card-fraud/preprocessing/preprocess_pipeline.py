import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import json

np.random.seed(42)

# ---- 1. Remover duplicatas (antes de qualquer split, para evitar leakage) ----
df_raw = pd.read_csv('/sessions/magical-busy-rubin/mnt/uploads/creditcard.csv')
n_before = len(df_raw)
df = df_raw.drop_duplicates().reset_index(drop=True)
n_after = len(df)
print(f"1) Duplicatas removidas: {n_before - n_after} ({n_before} -> {n_after})")
print(f"   Fraudes antes: {(df_raw['Class']==1).sum()} | depois: {(df['Class']==1).sum()}")

# ---- 3. Split estratificado 70/15/15 (feito antes do z-score para evitar leakage de estatisticas) ----
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print("\n3) Split estratificado 70/15/15:")
for name, Xs, ys in [('treino', X_train, y_train), ('validação', X_val, y_val), ('teste', X_test, y_test)]:
    n = len(ys)
    nf = int((ys==1).sum())
    print(f"   {name:10s}: {Xs.shape} | fraudes: {nf} ({nf/n*100:.4f}%)")

# ---- 2. Z-score em Amount e Time (fit SOMENTE no treino, aplicado nos 3 conjuntos) ----
# Nota: ajustei a ordem para fitar o scaler apenas com estatisticas do treino.
# Se o z-score fosse calculado com media/desvio do dataset inteiro (antes do split),
# as estatisticas de validacao/teste vazariam para o treino - o mesmo tipo de
# data leakage que a remocao de duplicatas antes do split ja evita.
scaler = StandardScaler()
scaler.fit(X_train[['Amount', 'Time']])

for Xs in [X_train, X_val, X_test]:
    Xs[['Amount_z', 'Time_z']] = scaler.transform(Xs[['Amount', 'Time']])

print(f"\n2) Z-score aplicado em Amount e Time (fit no treino):")
print(f"   Amount -> media treino: {scaler.mean_[0]:.2f}, desvio: {scaler.scale_[0]:.2f}")
print(f"   Time   -> media treino: {scaler.mean_[1]:.2f}, desvio: {scaler.scale_[1]:.2f}")

# reordenar colunas: manter V1-V28, Amount_z, Time_z (dropar Amount/Time originais)
v_cols = [c for c in X.columns if c.startswith('V')]
final_cols = v_cols + ['Amount_z', 'Time_z']

def finalize(Xs, ys):
    out = Xs[final_cols].copy()
    out['Class'] = ys.values
    return out.reset_index(drop=True)

train_df = finalize(X_train, y_train)
val_df = finalize(X_val, y_val)
test_df = finalize(X_test, y_test)

# ---- 4. class_weight (sem oversampling/undersampling) ----
classes = np.array([0, 1])
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
print(f"\n4) class_weight (calculado a partir do treino): {class_weight_dict}")

# ---- Confirmacao de nao sobreposicao entre conjuntos ----
# indices originais (do df deduplicado) sao unicos por particao (train_test_split particiona sem reposicao)
idx_train, idx_val, idx_test = set(X_train.index), set(X_val.index), set(X_test.index)
overlap_tv = idx_train & idx_val
overlap_tt = idx_train & idx_test
overlap_vt = idx_val & idx_test
print(f"\nConfirmação de sobreposição entre conjuntos (por índice original):")
print(f"   treino ∩ validação: {len(overlap_tv)} linhas")
print(f"   treino ∩ teste:     {len(overlap_tt)} linhas")
print(f"   validação ∩ teste:  {len(overlap_vt)} linhas")
print(f"   Total de linhas nos 3 conjuntos: {len(idx_train)+len(idx_val)+len(idx_test)} (esperado: {len(df)})")

# checagem extra: nenhuma linha (todas as colunas originais) aparece em mais de um conjunto
all_idx = pd.concat([pd.Series(list(idx_train)), pd.Series(list(idx_val)), pd.Series(list(idx_test))])
print(f"   Índices duplicados entre conjuntos: {all_idx.duplicated().sum()}")

# ---- Salvar ----
out_dir = 'tt-docs/projetos/fraude/preprocessing'
train_df.to_csv(f'{out_dir}/train.csv', index=False)
val_df.to_csv(f'{out_dir}/val.csv', index=False)
test_df.to_csv(f'{out_dir}/test.csv', index=False)

meta = {
    'n_before_dedup': int(n_before),
    'n_after_dedup': int(n_after),
    'duplicates_removed': int(n_before - n_after),
    'scaler_amount_mean': float(scaler.mean_[0]),
    'scaler_amount_std': float(scaler.scale_[0]),
    'scaler_time_mean': float(scaler.mean_[1]),
    'scaler_time_std': float(scaler.scale_[1]),
    'class_weight': class_weight_dict,
    'splits': {
        'train': {'n': int(len(train_df)), 'n_fraud': int((train_df.Class==1).sum())},
        'val': {'n': int(len(val_df)), 'n_fraud': int((val_df.Class==1).sum())},
        'test': {'n': int(len(test_df)), 'n_fraud': int((test_df.Class==1).sum())},
    },
    'overlap_check': {
        'train_val': len(overlap_tv), 'train_test': len(overlap_tt), 'val_test': len(overlap_vt),
        'duplicated_indices_across_splits': int(all_idx.duplicated().sum())
    }
}
with open(f'{out_dir}/preprocessing_meta.json', 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nArquivos salvos em {out_dir}/: train.csv, val.csv, test.csv, preprocessing_meta.json")
