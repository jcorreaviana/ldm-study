"""
REGRESSÃO LINEAR DO ZERO — v2 (otimizada)
==========================================
Evolução da v1 com as seguintes melhorias:

    1. Divisão treino / validação / teste  (70% / 15% / 15%)
    2. Early Stopping com patience
    3. Monitoramento do gap treino vs validação (detecção de overfitting)
    4. R² calculado no conjunto de teste ao final
    5. Relatório comparativo v1 vs v2

Conceitos adicionados em relação à v1:
    - MSE de validação  →  usado pelo Early Stopping
    - MSE de teste      →  medida final de qualidade
    - patience          →  quantas épocas sem melhora antes de parar
    - R²                →  "quanto da variação real o modelo explica?" (0 a 1)
"""

import random

# ============================================================
# DATASET (30 exemplos — suficiente para divisão treino/val/teste)
# ============================================================
dataset = [
    (1.55, 50.0), (1.57, 52.0), (1.60, 55.0), (1.62, 56.5),
    (1.63, 57.0), (1.65, 59.0), (1.67, 61.0), (1.68, 62.0),
    (1.70, 64.0), (1.71, 65.0), (1.72, 66.0), (1.73, 67.5),
    (1.75, 70.0), (1.76, 71.0), (1.77, 72.0), (1.78, 73.0),
    (1.80, 75.0), (1.81, 76.5), (1.82, 78.0), (1.83, 79.0),
    (1.85, 81.0), (1.86, 82.0), (1.87, 83.0), (1.88, 84.0),
    (1.90, 85.0), (1.92, 87.0), (1.95, 90.0), (1.97, 92.0),
    (2.00, 95.0), (2.02, 97.0),
]

# ============================================================
# DIVISÃO TREINO / VALIDAÇÃO / TESTE
# ============================================================
# treino     70%  →  o modelo aprende aqui
# validação  15%  →  Early Stopping monitora aqui
# teste      15%  →  avaliação final, usado UMA só vez
random.seed(42)
dados = dataset[:]
random.shuffle(dados)

n_total  = len(dados)
n_treino = int(n_total * 0.70)
n_val    = int(n_total * 0.15)

treino    = dados[:n_treino]
validacao = dados[n_treino:n_treino + n_val]
teste     = dados[n_treino + n_val:]

print("=" * 60)
print("DIVISÃO DO DATASET")
print("=" * 60)
print(f"  total:     {n_total} exemplos")
print(f"  treino:    {len(treino)} exemplos  (70%)")
print(f"  validação: {len(validacao)} exemplos  (15%)")
print(f"  teste:     {len(teste)} exemplos  (15%)")

# ============================================================
# NORMALIZAÇÃO — baseada APENAS no treino
# ============================================================
# Importante: a média é calculada só com dados de treino
# para não "contaminar" o modelo com informações de val/teste
media_x = sum(x for x, _ in treino) / len(treino)

def normalizar(x):
    return x - media_x

# ============================================================
# MODELO — f(x) = a*x + b
# ============================================================
def modelo(x, a, b):
    return a * x + b

# ============================================================
# FUNÇÃO DE PERDA — MSE
# L = (1/n) * Σ (f(xᵢ) - yᵢ)²
# ============================================================
def mse(dados, a, b):
    total = 0.0
    for x, y in dados:
        y_prev = modelo(normalizar(x), a, b)
        total += (y_prev - y) ** 2
    return total / len(dados)

# ============================================================
# GRADIENTES — ∂L/∂a e ∂L/∂b
# (calculados APENAS com dados de treino)
# ============================================================
def gradientes(dados_treino, a, b):
    dL_da = 0.0
    dL_db = 0.0
    for x, y in dados_treino:
        xn = normalizar(x)
        e  = modelo(xn, a, b) - y
        dL_da += 2 * e * xn
        dL_db += 2 * e
    n = len(dados_treino)
    return dL_da / n, dL_db / n

# ============================================================
# R² — coeficiente de determinação
# Responde: "quanto da variação real o modelo consegue explicar?"
#
# R² = 1 - (Σ(y_prev - y_real)²) / (Σ(y_real - média_y)²)
#
# R² = 1.0  →  modelo perfeito
# R² = 0.0  →  modelo não melhor que prever sempre a média
# R² < 0.0  →  modelo pior que prever sempre a média
# ============================================================
def r2(dados, a, b):
    ys = [y for _, y in dados]
    media_y = sum(ys) / len(ys)
    ss_res = sum((modelo(normalizar(x), a, b) - y) ** 2 for x, y in dados)
    ss_tot = sum((y - media_y) ** 2 for y in ys)
    return 1 - ss_res / ss_tot

# ============================================================
# TREINAMENTO COM EARLY STOPPING
# ============================================================
a  = 1.0
b  = 1.0
lr = 0.1

patience          = 50
max_epocas        = 5000
min_delta         = 1e-4

melhor_val        = float('inf')
patience_contador = 0
epoca_parada      = 0

hist_treino = []
hist_val    = []

print("\n" + "=" * 60)
print("TREINANDO com Early Stopping")
print(f"patience = {patience} épocas  |  lr = {lr}")
print("=" * 60)
print(f"{'Época':>6}  {'MSE treino':>12}  {'MSE val':>10}  {'gap':>10}  {'patience':>10}")
print("-" * 60)

for epoca in range(max_epocas):

    # 1. gradiente descendente (só com dados de treino)
    dL_da, dL_db = gradientes(treino, a, b)
    a = a - lr * dL_da
    b = b - lr * dL_db

    # 2. calcula perdas
    perda_treino = mse(treino, a, b)
    perda_val    = mse(validacao, a, b)
    gap          = perda_val - perda_treino

    hist_treino.append(perda_treino)
    hist_val.append(perda_val)

    # 3. Early Stopping — monitora MSE de validação
    if perda_val < melhor_val - min_delta:
        melhor_val        = perda_val
        patience_contador = 0
    else:
        patience_contador += 1

    if epoca % 100 == 0 or patience_contador == patience:
        print(
            f"{epoca:>6}  {perda_treino:>12.4f}  {perda_val:>10.4f}"
            f"  {gap:>10.4f}  {patience_contador:>10}/{patience}"
        )

    if patience_contador >= patience:
        epoca_parada = epoca
        print(f"\n  Early Stopping ativado na época {epoca}!")
        break
else:
    epoca_parada = max_epocas

# ============================================================
# AVALIAÇÃO FINAL NO CONJUNTO DE TESTE
# ============================================================
mse_teste = mse(teste, a, b)
r2_teste  = r2(teste, a, b)

print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)
print(f"  Função aprendida: f(x) = {a:.4f} * (x - {media_x:.4f}) + {b:.4f}")
print(f"  Épocas executadas: {epoca_parada}  (limite: {max_epocas})")
print(f"  MSE treino:        {hist_treino[-1]:.4f}")
print(f"  MSE validação:     {hist_val[-1]:.4f}")
print(f"  MSE teste:         {mse_teste:.4f}")
print(f"  R² (teste):        {r2_teste:.4f}  {'✓ bom' if r2_teste > 0.9 else '⚠ revisar'}")

# ============================================================
# INFERÊNCIAS
# ============================================================
print("\n" + "=" * 60)
print("INFERÊNCIAS (alturas fora do dataset de treino)")
print("=" * 60)
for altura in [1.58, 1.69, 1.79, 1.93]:
    peso = modelo(normalizar(altura), a, b)
    print(f"  altura = {altura:.2f} m  →  peso previsto = {peso:.1f} kg")

# ============================================================
# RELATÓRIO COMPARATIVO v1 vs v2
# ============================================================
print("\n" + "=" * 60)
print("COMPARATIVO v1 vs v2")
print("=" * 60)
print(f"  {'':30}  {'v1':>12}  {'v2':>12}")
print(f"  {'épocas executadas':30}  {'1000':>12}  {epoca_parada:>12}")
print(f"  {'critério de parada':30}  {'fixo':>12}  {'early stop':>12}")
print(f"  {'divisão treino/val/teste':30}  {'não':>12}  {'sim':>12}")
print(f"  {'R² calculado':30}  {'não':>12}  {r2_teste:>12.4f}")
print(f"  {'gap monitorado':30}  {'não':>12}  {'sim':>12}")
print(f"  {'risco de overfitting':30}  {'alto':>12}  {'controlado':>12}")