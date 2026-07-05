"""
EXEMPLO DIDÁTICO: Função Altura -> Peso usando micrograd
=========================================================

Conceito matemático:
    f: X -> Y
    onde X (domínio)  = alturas possíveis
         Y (imagem)    = pesos possíveis

Queremos ENCONTRAR os parâmetros 'a' e 'b' de uma função linear:
    f(x) = a*x + b

de forma que f(altura) se aproxime do peso real, usando apenas
exemplos observados (dataset) e gradiente descendente, calculado
via autodiferenciação (regra da cadeia) com a biblioteca micrograd.

Ao final, gera um gráfico PNG mostrando:
    - os dados reais (pontos)
    - a reta aprendida pelo modelo
    - a evolução da perda (erro) ao longo do treinamento
"""

import matplotlib.pyplot as plt
from micrograd.engine import Value

# ---------------------------------------------------------
# 1. O DATASET (pares conhecidos de domínio -> imagem)
# ---------------------------------------------------------
dataset = [
    (1.60, 55.0),
    (1.65, 59.0),
    (1.70, 64.0),
    (1.75, 70.0),
    (1.80, 75.0),
    (1.85, 81.0),
    (1.90, 85.0),
]

print("=" * 55)
print("DATASET (domínio X -> imagem Y)")
print("=" * 55)
for x, y in dataset:
    print(f"  altura = {x:.2f} m   ->   peso = {y:.1f} kg")

# ---------------------------------------------------------
# 2. OS PARÂMETROS DA FUNÇÃO (o que o modelo vai aprender)
# ---------------------------------------------------------
a = Value(1.0)
b = Value(1.0)

print("\n" + "=" * 55)
print("PARÂMETROS INICIAIS (chutados, ainda errados)")
print("=" * 55)
print(f"  a = {a.data:.4f}")
print(f"  b = {b.data:.4f}")

# ---------------------------------------------------------
# NOTA DIDÁTICA: por que normalizar?
# ---------------------------------------------------------
# Como 'a' multiplica valores em metros (~1.6 a 1.9) e 'b' é somado
# diretamente, o gradiente de 'a' fica MUITO menor que o de 'b'.
# Isso faz o treinamento "andar torto" e demorar para convergir.
# Solução padrão em ML: normalizar a entrada, equilibrando a
# magnitude dos gradientes de a e b.
media_x = sum(x for x, _ in dataset) / len(dataset)


def normalizar(x):
    return x - media_x


def f(x_normalizado):
    """A função que estamos tentando ajustar: f(x) = a*x + b"""
    return a * x_normalizado + b


# ---------------------------------------------------------
# 3. TREINAMENTO: forward -> erro -> backward -> atualizar
# ---------------------------------------------------------
learning_rate = 0.1
epocas = 1000

historico_perda = []
historico_a = []
historico_b = []

print("\n" + "=" * 55)
print("TREINANDO (ajustando a e b para minimizar o erro)")
print("=" * 55)

for epoca in range(epocas):
    # --- FORWARD PASS ---
    perda_total = Value(0.0)
    for x, y_real in dataset:
        y_previsto = f(normalizar(x))
        erro = (y_previsto - y_real) ** 2  # erro quadrático
        perda_total = perda_total + erro

    perda_total = perda_total / len(dataset)  # erro médio

    # --- BACKWARD PASS ---
    a.grad = 0.0
    b.grad = 0.0
    perda_total.backward()  # REGRA DA CADEIA calcula d(perda)/da e d(perda)/db

    # --- ATUALIZAÇÃO (gradiente descendente) ---
    a.data -= learning_rate * a.grad
    b.data -= learning_rate * b.grad

    historico_perda.append(perda_total.data)
    historico_a.append(a.data)
    historico_b.append(b.data)

    if epoca % 100 == 0 or epoca == epocas - 1:
        print(
            f"  época {epoca:4d}  |  perda = {perda_total.data:8.4f}  "
            f"|  a = {a.data:8.4f}  b = {b.data:8.4f}"
        )

# ---------------------------------------------------------
# 4. RESULTADO FINAL
# ---------------------------------------------------------
print("\n" + "=" * 55)
print("FUNÇÃO APRENDIDA (com x normalizado: x' = x - média)")
print("=" * 55)
print(f"  f(x) = {a.data:.4f} * (x - {media_x:.4f}) + {b.data:.4f}")

print("\n" + "=" * 55)
print("TESTANDO com alturas novas (fora do dataset original)")
print("=" * 55)
testes = [1.55, 1.72, 1.95]
previsoes_teste = []
for altura_teste in testes:
    peso_previsto = f(normalizar(altura_teste))
    previsoes_teste.append(peso_previsto.data)
    print(f"  altura = {altura_teste:.2f} m   ->   peso previsto = {peso_previsto.data:.1f} kg")

# ===========================================================
# 5. VISUALIZAÇÃO — gráfico PNG com apelo visual forte
# ===========================================================
plt.style.use("dark_background")

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]}
)
fig.patch.set_facecolor("#0d1117")

# --- Painel 1: dados reais x reta ajustada ---
ax1.set_facecolor("#0d1117")

xs_dataset = [x for x, _ in dataset]
ys_dataset = [y for _, y in dataset]

# pontos reais com glow (camadas de pontos translúcidos por trás)
for size, alpha in [(900, 0.08), (500, 0.15), (220, 0.35)]:
    ax1.scatter(xs_dataset, ys_dataset, s=size, color="#ff6b6b", alpha=alpha, zorder=2)
ax1.scatter(
    xs_dataset, ys_dataset, s=90, color="#ff4757", edgecolor="white",
    linewidth=1.2, zorder=3, label="Dados reais"
)

# reta aprendida, com glow também
x_linha = [1.5, 2.0]
y_linha = [f(normalizar(x)).data for x in x_linha]
for lw, alpha in [(10, 0.10), (6, 0.20), (3, 0.45)]:
    ax1.plot(x_linha, y_linha, color="#00d4ff", linewidth=lw, alpha=alpha, zorder=1)
ax1.plot(
    x_linha, y_linha, color="#00d4ff", linewidth=2.5, zorder=4,
    label=f"f(x) = {a.data:.1f}·(x-{media_x:.2f}) + {b.data:.1f}"
)

# previsões em pontos novos (fora do dataset)
ax1.scatter(
    testes, previsoes_teste, s=140, marker="D", color="#feca57",
    edgecolor="white", linewidth=1.2, zorder=5, label="Previsões (novos pontos)"
)

ax1.set_xlabel("Altura (m) — domínio", fontsize=12, color="#c9d1d9")
ax1.set_ylabel("Peso (kg) — imagem", fontsize=12, color="#c9d1d9")
ax1.set_title(
    "Função aprendida: f(altura) → peso", fontsize=15, color="white", pad=14, weight="bold"
)
ax1.legend(loc="upper left", fontsize=10, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
ax1.grid(True, color="#30363d", linewidth=0.6, alpha=0.6)
ax1.tick_params(colors="#8b949e")
for spine in ax1.spines.values():
    spine.set_color("#30363d")

# --- Painel 2: evolução da perda durante o treinamento ---
ax2.set_facecolor("#0d1117")
ax2.plot(historico_perda, color="#ff6b6b", linewidth=2.2)
ax2.fill_between(range(len(historico_perda)), historico_perda, color="#ff6b6b", alpha=0.15)
ax2.set_yscale("log")
ax2.set_xlabel("Época", fontsize=12, color="#c9d1d9")
ax2.set_ylabel("Perda (escala log)", fontsize=12, color="#c9d1d9")
ax2.set_title("Erro caindo durante o treinamento", fontsize=15, color="white", pad=14, weight="bold")
ax2.grid(True, color="#30363d", linewidth=0.6, alpha=0.6)
ax2.tick_params(colors="#8b949e")
for spine in ax2.spines.values():
    spine.set_color("#30363d")

fig.suptitle(
    "Regressão linear com autodiferenciação (micrograd)",
    fontsize=17, color="white", weight="bold", y=1.02
)

plt.tight_layout()
plt.savefig("regressao_altura_peso.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nGráfico salvo em: regressao_altura_peso.png")