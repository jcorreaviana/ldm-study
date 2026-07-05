"""
COMPARATIVO: Do Zero vs Micrograd
==================================
O mesmo problema — regressão linear altura → peso —
implementado de duas formas:

    v1 (do zero)   →  você escreve TUDO manualmente
    v2 (micrograd) →  a biblioteca cuida da parte de gradientes

O objetivo é mostrar o que o micrograd abstrai e qual o
diferencial que ele traz em relação à implementação manual.
"""

# ============================================================
# DATASET (igual nos dois)
# ============================================================
dataset = [
    (1.60, 55.0), (1.65, 59.0), (1.70, 64.0),
    (1.75, 70.0), (1.80, 75.0), (1.85, 81.0), (1.90, 85.0),
]
media_x = sum(x for x, _ in dataset) / len(dataset)

# ============================================================
# ============================================================
# VERSÃO 1 — DO ZERO (sem biblioteca)
# ============================================================
# ============================================================
print("=" * 60)
print("VERSÃO 1 — DO ZERO (sem biblioteca)")
print("=" * 60)

# parâmetros
a1 = 1.0
b1 = 1.0

def modelo_v1(x, a, b):
    return a * x + b

# GRADIENTES CALCULADOS À MÃO usando regra da cadeia
# ∂L/∂a = (1/n) * Σ 2 * eᵢ * xᵢ
# ∂L/∂b = (1/n) * Σ 2 * eᵢ
def gradientes_v1(a, b):
    dL_da, dL_db = 0.0, 0.0
    for x, y in dataset:
        xn = x - media_x
        e  = modelo_v1(xn, a, b) - y
        dL_da += 2 * e * xn   # ← você deriva isso manualmente
        dL_db += 2 * e         # ← você deriva isso manualmente
    n = len(dataset)
    return dL_da / n, dL_db / n

lr = 0.1
for epoca in range(1000):
    dL_da, dL_db = gradientes_v1(a1, b1)
    a1 = a1 - lr * dL_da
    b1 = b1 - lr * dL_db

print(f"  a = {a1:.4f}   b = {b1:.4f}")
print(f"  previsão para 1.72m: {modelo_v1(1.72 - media_x, a1, b1):.1f} kg")


# ============================================================
# ============================================================
# VERSÃO 2 — COM MICROGRAD
# ============================================================
# ============================================================
print("\n" + "=" * 60)
print("VERSÃO 2 — COM MICROGRAD")
print("=" * 60)

from micrograd.engine import Value

# parâmetros agora são objetos Value — rastreiam operações
a2 = Value(1.0)
b2 = Value(1.0)

def modelo_v2(x, a, b):
    return a * x + b   # mesma fórmula — mas agora constrói grafo

lr = 0.1
for epoca in range(1000):

    # FORWARD PASS — micrograd rastreia cada operação em um grafo
    perda = Value(0.0)
    for x, y in dataset:
        xn = x - media_x
        y_prev = modelo_v2(xn, a2, b2)
        erro   = (y_prev - y) ** 2
        perda  = perda + erro
    perda = perda / len(dataset)

    # BACKWARD PASS — regra da cadeia calculada AUTOMATICAMENTE
    # percorre o grafo de trás para frente e preenche .grad
    a2.grad = 0.0
    b2.grad = 0.0
    perda.backward()   # ← UMA linha substitui toda a função gradientes_v1()

    # atualização — igual à v1
    a2.data -= lr * a2.grad
    b2.data -= lr * b2.grad

print(f"  a = {a2.data:.4f}   b = {b2.data:.4f}")
print(f"  previsão para 1.72m: {modelo_v2(1.72 - media_x, a2, b2).data:.1f} kg")


# ============================================================
# COMPARATIVO DIRETO
# ============================================================
print("\n" + "=" * 60)
print("COMPARATIVO")
print("=" * 60)
print(f"  {'':35}  {'do zero':>10}  {'micrograd':>10}")
print(f"  {'resultado a':35}  {a1:>10.4f}  {a2.data:>10.4f}")
print(f"  {'resultado b':35}  {b1:>10.4f}  {b2.data:>10.4f}")
print()
print("  O QUE MUDA ENTRE AS DUAS VERSÕES:")
print()
print("  do zero:")
print("    → você deriva ∂L/∂a e ∂L/∂b manualmente")
print("    → qualquer mudança na função de perda exige")
print("      rederivação manual das fórmulas")
print("    → funciona, mas não escala para modelos complexos")
print()
print("  micrograd:")
print("    → você só define o forward pass (a conta)")
print("    → .backward() calcula todos os gradientes")
print("      automaticamente via regra da cadeia")
print("    → adicionar relu, camadas, novas operações?")
print("      zero mudança no código de gradientes")


# ============================================================
# O DIFERENCIAL REAL DO MICROGRAD
# ============================================================
print("\n" + "=" * 60)
print("O DIFERENCIAL REAL")
print("=" * 60)
print("""
  PROBLEMA com a versão do zero ao escalar:

  Imagine que você quer mudar a função de perda de MSE para
  algo mais complexo, ou adicionar uma função de ativação relu:

    do zero:
      → precisa rederivada todas as fórmulas de gradiente
      → para cada nova operação, nova derivada manual
      → em redes profundas com milhões de parâmetros: inviável

    micrograd:
      → muda só o forward pass
      → .backward() continua funcionando automaticamente
      → é exatamente assim que PyTorch e TensorFlow funcionam

  EXEMPLO: adicionando relu na v2 não muda nada no treinamento:

    y_prev = modelo_v2(xn, a2, b2).relu()  ← só essa linha muda
    # o .backward() ainda funciona corretamente

  Isso é o que torna o micrograd poderoso para fins didáticos:
  ele é pequeno o suficiente para você ler o código (~100 linhas),
  mas implementa o mesmo mecanismo do PyTorch.
""")