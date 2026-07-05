"""
REGRESSÃO LINEAR COM MÚLTIPLAS FEATURES — do zero
===================================================
Evolução do modelo de 1 feature para 2 features.

Modelo anterior (1 feature):
    f(x) = a * x + b
    entrada: altura
    saída:   peso

Modelo atual (2 features):
    f(X) = a1 * altura + a2 * idade + b
    entradas: [altura, idade]
    saída:    peso

Em notação vetorial (produto escalar):
    f(X) = W · X + b
    onde W = [a1, a2]  e  X = [altura, idade]

Novidade em relação à v1:
    Normalização por z-score — necessária quando features têm
    escalas diferentes (altura em metros vs idade em anos).
    Sem isso, o gradiente da idade explode e o treinamento diverge.

    z-score: x' = (x - média) / desvio_padrão
"""

# ============================================================
# DATASET
# cada exemplo = ([altura, idade], peso_real)
# ============================================================
dataset = [
    ([1.60, 20], 55.0),
    ([1.65, 25], 59.0),
    ([1.70, 30], 64.0),
    ([1.75, 22], 70.0),
    ([1.80, 35], 75.0),
    ([1.85, 28], 81.0),
    ([1.90, 40], 85.0),
    ([1.70, 45], 68.0),
    ([1.75, 50], 73.0),
    ([1.80, 55], 78.0),
]

n          = len(dataset)
n_features = len(dataset[0][0])

# ============================================================
# NORMALIZAÇÃO Z-SCORE — média e desvio padrão por feature
# ============================================================
# Por que z-score aqui e não só subtrair a média (como na v1)?
# Na v1, todas as entradas eram alturas — mesma escala.
# Aqui, altura (~1.75) e idade (~35) têm magnitudes muito diferentes.
# Sem normalizar pelo desvio padrão, o gradiente de idade fica
# ~20x maior que o de altura — o treinamento diverge (explode).
#
# z-score garante que todas as features fiquem na mesma escala:
# média = 0, desvio padrão = 1 para cada feature.

medias = [
    sum(dataset[i][0][j] for i in range(n)) / n
    for j in range(n_features)
]

desvios = [
    (sum((dataset[i][0][j] - medias[j]) ** 2 for i in range(n)) / n) ** 0.5
    for j in range(n_features)
]

print("=" * 60)
print("NORMALIZAÇÃO Z-SCORE POR FEATURE")
print("=" * 60)
print(f"  altura:  média = {medias[0]:.4f} m      desvio = {desvios[0]:.4f}")
print(f"  idade:   média = {medias[1]:.4f} anos   desvio = {desvios[1]:.4f}")

def normalizar(X):
    """z-score: x' = (x - média) / desvio"""
    return [(X[j] - medias[j]) / desvios[j] for j in range(n_features)]

# ============================================================
# MODELO — f(X) = W · X + b  (produto escalar)
# ============================================================
def modelo(X_norm, W, b):
    return sum(W[j] * X_norm[j] for j in range(n_features)) + b

# ============================================================
# FUNÇÃO DE PERDA — MSE
# ============================================================
def perda(W, b):
    total = 0.0
    for X, y in dataset:
        y_prev = modelo(normalizar(X), W, b)
        total += (y_prev - y) ** 2
    return total / n

# ============================================================
# GRADIENTES — ∂L/∂W e ∂L/∂b
# ============================================================
# Regra da cadeia — igual à v1, mas para cada wⱼ:
#   ∂L/∂wⱼ = (1/n) * Σ 2 * eᵢ * Xᵢⱼ
#   ∂L/∂b  = (1/n) * Σ 2 * eᵢ
def gradientes(W, b):
    dL_dW = [0.0] * n_features
    dL_db = 0.0
    for X, y in dataset:
        X_norm = normalizar(X)
        e = modelo(X_norm, W, b) - y
        for j in range(n_features):
            dL_dW[j] += 2 * e * X_norm[j]
        dL_db += 2 * e
    dL_dW = [dL_dW[j] / n for j in range(n_features)]
    dL_db = dL_db / n
    return dL_dW, dL_db

# ============================================================
# TREINAMENTO
# ============================================================
W  = [1.0, 1.0]
b  = 1.0
lr = 0.1
epocas = 1000

print("\n" + "=" * 60)
print("TREINANDO")
print("=" * 60)
print(f"{'Época':>6}  {'Perda':>12}  {'W[0]':>10}  {'W[1]':>10}  {'b':>10}")
print("-" * 60)

for epoca in range(epocas):
    L = perda(W, b)
    dL_dW, dL_db = gradientes(W, b)

    for j in range(n_features):
        W[j] = W[j] - lr * dL_dW[j]
    b = b - lr * dL_db

    if epoca % 100 == 0 or epoca == epocas - 1:
        print(f"{epoca:>6}  {L:>12.4f}  {W[0]:>10.4f}  {W[1]:>10.4f}  {b:>10.4f}")

# ============================================================
# RESULTADO FINAL
# ============================================================
print("\n" + "=" * 60)
print("FUNÇÃO APRENDIDA")
print("=" * 60)
print(f"  f(X) = {W[0]:.4f} * (altura - {medias[0]:.2f}) / {desvios[0]:.4f}")
print(f"       + {W[1]:.4f} * (idade  - {medias[1]:.2f}) / {desvios[1]:.4f}")
print(f"       + {b:.4f}")
print()
print(f"  W[0] (peso da altura): {W[0]:.4f}")
print(f"  W[1] (peso da idade):  {W[1]:.4f}")
print(f"  b    (viés):           {b:.4f}")
print()
print("  Interpretação:")
print(f"  → altura tem peso {abs(W[0]):.2f} — {'mais' if abs(W[0]) > abs(W[1]) else 'menos'} influente que a idade")
print(f"  → idade  tem peso {abs(W[1]):.2f} — {'mais' if abs(W[1]) > abs(W[0]) else 'menos'} influente que a altura")

# ============================================================
# INFERÊNCIAS
# ============================================================
print("\n" + "=" * 60)
print("INFERÊNCIAS")
print("=" * 60)
testes = [
    ([1.65, 22], "jovem, baixa"),
    ([1.75, 35], "média, média"),
    ([1.90, 50], "alta, mais velho"),
]
for X, desc in testes:
    peso = modelo(normalizar(X), W, b)
    print(f"  altura={X[0]}m  idade={X[1]}a  ({desc:18})  →  peso previsto = {peso:.1f} kg")

# ============================================================
# COMPARATIVO: 1 feature vs 2 features
# ============================================================
print("\n" + "=" * 60)
print("O QUE MUDOU EM RELAÇÃO À V1 (1 feature)")
print("=" * 60)
print("""
  DATASET:
    v1:  (altura, peso)             →  1 número de entrada
    v2:  ([altura, idade], peso)    →  vetor de entrada R²

  NORMALIZAÇÃO:
    v1:  x' = x - média             →  subtrai a média
    v2:  x' = (x - média) / desvio  →  z-score (escala também)

  PARÂMETROS:
    v1:  a, b                       →  2 parâmetros
    v2:  W = [a1, a2], b            →  3 parâmetros

  MODELO:
    v1:  f(x) = a * x + b
    v2:  f(X) = W · X + b           (produto escalar)
              = a1*altura + a2*idade + b

  GRADIENTES:
    v1:  dL/da  (1 gradiente)
    v2:  dL/dW = [dL/da1, dL/da2]  (1 gradiente por feature)

  O QUE NÃO MUDOU:
    → regra da cadeia
    → gradiente descendente
    → MSE como função de perda
    → estrutura do loop de treinamento
""")