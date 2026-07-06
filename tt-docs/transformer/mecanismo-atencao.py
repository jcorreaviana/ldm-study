"""
MECANISMO DE ATENÇÃO — do zero
================================
Implementação completa do self-attention sem nenhuma biblioteca de ML.

As 4 etapas:
    1. criar Q, K, V multiplicando a entrada por matrizes de pesos
    2. calcular scores: Q · Kᵀ / √d
    3. aplicar softmax nos scores (transforma em probabilidades 0-1)
    4. calcular saída: scores · V

Contexto: detecção de fraude
    Cada token é um evento do cliente.
    O mecanismo de atenção aprende quais eventos
    são mais relevantes para avaliar o evento atual.
"""

import math
import random
random.seed(42)

# ============================================================
# CONFIGURAÇÃO
# ============================================================
n_tokens = 4   # número de eventos na sequência
d_model  = 8   # dimensão de cada token (vetor de 8 números)
d_k      = 4   # dimensão de Q e K (pode ser menor que d_model)

# ============================================================
# DATASET — sequência de 4 eventos do cliente
# ============================================================
# Cada evento é representado como um vetor de d_model números.
# Na prática esses vetores vêm de um embedding layer.
# Aqui usamos valores aleatórios para simplificar.

print("=" * 60)
print("SEQUÊNCIA DE EVENTOS (tokens)")
print("=" * 60)

eventos = [
    "login_normal",
    "compra_sao_paulo",
    "troca_senha_ip_estrangeiro",
    "compra_toquio_4h22"
]

# cada token = vetor de d_model números (simulando embedding)
random.seed(42)
tokens = [[random.gauss(0, 1) for _ in range(d_model)]
          for _ in range(n_tokens)]

for i, (evento, token) in enumerate(zip(eventos, tokens)):
    print(f"  token {i} — {evento}")
    print(f"    vetor: [{', '.join(f'{v:.2f}' for v in token)}]")

# ============================================================
# MATRIZES DE PESOS — aprendidas durante o treinamento
# ============================================================
# Wq, Wk, Wv são matrizes que transformam cada token em Q, K, V
# dimensão: [d_model × d_k]
# Na prática são parâmetros treináveis — aqui inicializamos aleatório

def init_matriz(linhas, colunas):
    escala = math.sqrt(2.0 / linhas)
    return [[random.gauss(0, escala) for _ in range(colunas)]
            for _ in range(linhas)]

Wq = init_matriz(d_model, d_k)   # [8 × 4]
Wk = init_matriz(d_model, d_k)   # [8 × 4]
Wv = init_matriz(d_model, d_k)   # [8 × 4]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def multiplicar_matrizes(A, B):
    """
    Multiplica duas matrizes A [m×n] e B [n×p].
    Resultado: C [m×p]
    C[i][j] = soma(A[i][k] * B[k][j] para cada k)
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def transpor(A):
    """
    Transposta de A [m×n] → Aᵀ [n×m]
    Troca linhas por colunas.
    """
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]

def softmax_linha(scores):
    """
    Aplica softmax em uma lista de scores.
    softmax(x) = e^xᵢ / Σ e^xⱼ
    Transforma scores em probabilidades que somam 1.
    """
    exp_scores = [math.exp(s) for s in scores]
    soma = sum(exp_scores)
    return [e / soma for e in exp_scores]

def softmax_matriz(M):
    """Aplica softmax em cada linha de uma matriz."""
    return [softmax_linha(linha) for linha in M]

# ============================================================
# ETAPA 1 — CRIAR Q, K, V
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 1 — Criando Q, K, V")
print("=" * 60)
print(f"  tokens: [{n_tokens} × {d_model}]")
print(f"  Wq, Wk, Wv: [{d_model} × {d_k}]")
print(f"  Q, K, V: [{n_tokens} × {d_k}]")
print()

# Q = tokens · Wq  →  [4 × 8] · [8 × 4] = [4 × 4]
# K = tokens · Wk  →  [4 × 8] · [8 × 4] = [4 × 4]
# V = tokens · Wv  →  [4 × 8] · [8 × 4] = [4 × 4]

Q = multiplicar_matrizes(tokens, Wq)
K = multiplicar_matrizes(tokens, Wk)
V = multiplicar_matrizes(tokens, Wv)

print("  Q (Query) — o que cada token está procurando:")
for i, (evento, q) in enumerate(zip(eventos, Q)):
    print(f"    token {i} ({evento[:20]:20}): [{', '.join(f'{v:.2f}' for v in q)}]")

# ============================================================
# ETAPA 2 — CALCULAR SCORES: Q · Kᵀ / √d
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 2 — Calculando scores de atenção")
print("=" * 60)
print(f"  Q · Kᵀ: [{n_tokens} × {d_k}] · [{d_k} × {n_tokens}] = [{n_tokens} × {n_tokens}]")
print(f"  dividindo por √d = √{d_k} = {math.sqrt(d_k):.2f}")
print()

Kt = transpor(K)
scores_raw = multiplicar_matrizes(Q, Kt)

# divide por √d_k para estabilizar os gradientes
# sem isso, scores grandes → softmax muito extremo → gradientes desaparecem
escala = math.sqrt(d_k)
scores_scaled = [[s / escala for s in linha] for linha in scores_raw]

print("  scores (Q · Kᵀ / √d) — relevância entre cada par de tokens:")
print(f"  {'':30}", end="")
for evento in eventos:
    print(f"  {evento[:12]:12}", end="")
print()
for i, (evento, linha) in enumerate(zip(eventos, scores_scaled)):
    print(f"  token {i} ({evento[:20]:20}):", end="")
    for s in linha:
        print(f"  {s:12.3f}", end="")
    print()

# ============================================================
# ETAPA 3 — SOFTMAX: transforma scores em probabilidades
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 3 — Softmax (scores → probabilidades de atenção)")
print("=" * 60)

atencao = softmax_matriz(scores_scaled)

print("  pesos de atenção — cada linha soma 1.0:")
print(f"  {'':30}", end="")
for evento in eventos:
    print(f"  {evento[:12]:12}", end="")
print()
for i, (evento, linha) in enumerate(zip(eventos, atencao)):
    print(f"  token {i} ({evento[:20]:20}):", end="")
    for a in linha:
        print(f"  {a:12.3f}", end="")
    print()
    soma = sum(linha)
    print(f"  {'soma':52} {soma:.3f}")

# ============================================================
# ETAPA 4 — SAÍDA: atenção · V
# ============================================================
print("\n" + "=" * 60)
print("ETAPA 4 — Saída: atenção · V")
print("=" * 60)
print(f"  atenção [{n_tokens} × {n_tokens}] · V [{n_tokens} × {d_k}] = saída [{n_tokens} × {d_k}]")
print()

saida = multiplicar_matrizes(atencao, V)

print("  saída — representação enriquecida de cada token:")
for i, (evento, s) in enumerate(zip(eventos, saida)):
    print(f"  token {i} ({evento[:20]:20}): [{', '.join(f'{v:.3f}' for v in s)}]")

# ============================================================
# INTERPRETAÇÃO
# ============================================================
print("\n" + "=" * 60)
print("INTERPRETAÇÃO — qual evento recebeu mais atenção?")
print("=" * 60)

# token da compra em tóquio (token 3) — quais eventos ele mais considera?
token_alvo = 3
pesos = atencao[token_alvo]
print(f"\n  Quando o modelo avalia '{eventos[token_alvo]}':")
print(f"  ele distribui atenção assim:")
for i, (evento, peso) in enumerate(zip(eventos, pesos)):
    barra = "█" * int(peso * 40)
    print(f"    {evento[:30]:30}  {peso:.3f}  {barra}")

# ============================================================
# RESUMO DO FLUXO
# ============================================================
print("\n" + "=" * 60)
print("RESUMO DO FLUXO")
print("=" * 60)
print("""
  entrada (tokens)   [4 × 8]
       ↓ × Wq, Wk, Wv
  Q, K, V            [4 × 4]
       ↓ Q · Kᵀ / √d
  scores             [4 × 4]   ← relevância entre cada par de tokens
       ↓ softmax
  atenção            [4 × 4]   ← probabilidades (cada linha soma 1)
       ↓ × V
  saída              [4 × 4]   ← tokens enriquecidos com contexto
""")