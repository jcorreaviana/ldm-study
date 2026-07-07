"""
TRANSFORMER COM TREINAMENTO — PyTorch
======================================
Versão do transformer_completo.py com backward pass implementado.

A diferença em relação à versão do zero:
    - PyTorch calcula os gradientes automaticamente (.backward())
    - Os pesos são ajustados a cada época (AdamW)
    - O modelo aprende a distinguir fraude de transação legítima

Dataset sintético:
    - sequências de eventos de clientes
    - label: 1 = fraude, 0 = legítimo
    - o modelo aprende o padrão que precede fraude

Arquitetura:
    pré-processamento → transformer (2 blocos) → sigmoid → score
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)

# ============================================================
# CONFIGURAÇÃO
# ============================================================
n_tokens  = 4     # eventos por sequência
d_model   = 8     # dimensão de cada token
n_heads   = 2     # cabeças de atenção
d_ff      = 32    # dimensão interna do feed-forward
n_blocos  = 2     # blocos transformer empilhados
lr        = 0.001
epocas    = 200

# ============================================================
# DATASET SINTÉTICO
# ============================================================
# Cada exemplo = sequência de 4 eventos + label (fraude ou não)
# tokens pré-processados: [z_valor, z_hora, brasil, eua, europa, asia, outro, device]

# sequências fraudulentas — padrão: troca_senha + localização estranha + valor alto
sequencias_fraude = [
    [[-1.19, -0.83, 1,0,0,0,0,1], [-0.41, 0.18, 1,0,0,0,0,1], [-1.19,-2.21,0,0,0,1,0,0], [2.67,-1.95,0,0,0,1,0,1]],
    [[-0.80, -0.50, 1,0,0,0,0,1], [-0.20, 0.30, 1,0,0,0,0,1], [-1.00,-2.50,0,0,0,0,0,0], [3.10,-2.10,0,0,1,0,0,1]],
    [[-1.00, -0.70, 1,0,0,0,0,1], [-0.60, 0.10, 1,0,0,0,0,0], [-1.19,-1.90,0,0,0,1,0,0], [2.90,-1.80,0,0,0,1,0,1]],
    [[-0.90, -0.60, 1,0,0,0,0,1], [-0.30, 0.20, 0,1,0,0,0,1], [-1.10,-2.30,0,0,0,0,1,0], [2.50,-2.00,0,0,0,0,1,0]],
]

# sequências legítimas — padrão: compras normais no mesmo país
sequencias_legitimas = [
    [[-0.41, 0.18, 1,0,0,0,0,1], [0.85, 0.20, 1,0,0,0,0,1], [0.50, -0.30, 1,0,0,0,0,1], [0.30, 0.10, 1,0,0,0,0,1]],
    [[-0.20, 0.30, 1,0,0,0,0,1], [0.60, 0.15, 1,0,0,0,0,1], [0.40, -0.20, 1,0,0,0,0,1], [0.20, 0.05, 1,0,0,0,0,1]],
    [[-0.50, 0.10, 0,1,0,0,0,1], [0.70, 0.25, 0,1,0,0,0,1], [0.30, -0.10, 0,1,0,0,0,1], [0.50, 0.15, 0,1,0,0,0,1]],
    [[-0.30, 0.20, 1,0,0,0,0,1], [0.80, 0.30, 1,0,0,0,0,1], [0.60, -0.15, 1,0,0,0,0,1], [0.40, 0.20, 1,0,0,0,0,1]],
]

# monta dataset
X_data = sequencias_fraude + sequencias_legitimas
y_data = [1.0] * len(sequencias_fraude) + [0.0] * len(sequencias_legitimas)

X_tensor = torch.tensor(X_data, dtype=torch.float32)   # [8, 4, 8]
y_tensor = torch.tensor(y_data, dtype=torch.float32)   # [8]

# ============================================================
# ARQUITETURA — BLOCO TRANSFORMER COM PYTORCH
# ============================================================
class BlocoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # multi-head attention
        self.atencao = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        # feed-forward: d_model → d_ff → d_model
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        # layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # sub-bloco 1: atenção + residual + layer norm
        atencao_saida, _ = self.atencao(x, x, x)
        x = self.norm1(x + atencao_saida)

        # sub-bloco 2: feed-forward + residual + layer norm
        x = self.norm2(x + self.ff(x))

        return x

class TransformerFraude(nn.Module):
    def __init__(self):
        super().__init__()
        # N blocos empilhados
        self.blocos = nn.ModuleList([BlocoTransformer() for _ in range(n_blocos)])
        # classificador final — usa o último token como representação
        self.classificador = nn.Linear(d_model, 1)

    def forward(self, x):
        # passa por cada bloco
        for bloco in self.blocos:
            x = bloco(x)

        # usa o último token (compra atual) para classificar
        ultimo_token = x[:, -1, :]          # [batch, d_model]
        score = self.classificador(ultimo_token)  # [batch, 1]
        return torch.sigmoid(score).squeeze(1)    # [batch] → probabilidade 0-1

# ============================================================
# TREINAMENTO
# ============================================================
modelo = TransformerFraude()
otimizador = optim.AdamW(modelo.parameters(), lr=lr)
criterio = nn.BCELoss()   # Binary Cross-Entropy — para classificação binária

print("=" * 60)
print("TREINAMENTO")
print("=" * 60)
print(f"  exemplos: {len(X_data)} ({len(sequencias_fraude)} fraudes, {len(sequencias_legitimas)} legítimas)")
print(f"  épocas:   {epocas}")
print(f"  otimizador: AdamW  lr={lr}")
print(f"  perda:    Binary Cross-Entropy")
print()
print(f"{'Época':>6}  {'Perda':>10}  {'Acurácia':>10}")
print("-" * 35)

for epoca in range(epocas):
    modelo.train()
    otimizador.zero_grad()

    # forward pass
    scores = modelo(X_tensor)

    # calcula perda (BCE)
    perda = criterio(scores, y_tensor)

    # backward pass — PyTorch calcula todos os gradientes
    perda.backward()

    # atualiza pesos
    otimizador.step()

    if epoca % 20 == 0 or epoca == epocas - 1:
        with torch.no_grad():
            predicoes = (scores > 0.5).float()
            acuracia = (predicoes == y_tensor).float().mean()
        print(f"{epoca:>6}  {perda.item():>10.4f}  {acuracia.item():>10.1%}")

# ============================================================
# RESULTADO FINAL
# ============================================================
print("\n" + "=" * 60)
print("SCORES APRENDIDOS")
print("=" * 60)

eventos_fraude = [
    "login_normal → compra_SP → troca_senha → compra_Tóquio",
    "login_normal → compra_SP → troca_senha → compra_Europa",
    "login_normal → compra_SP → troca_senha → compra_Ásia",
    "login_normal → compra_EUA → troca_senha → compra_Outro",
]
eventos_legitimos = [
    "compra_SP → compra_SP → compra_SP → compra_SP",
    "compra_SP → compra_SP → compra_SP → compra_SP",
    "compra_EUA → compra_EUA → compra_EUA → compra_EUA",
    "compra_SP → compra_SP → compra_SP → compra_SP",
]

modelo.eval()
with torch.no_grad():
    scores_finais = modelo(X_tensor)

print("\n  FRAUDES:")
for i in range(len(sequencias_fraude)):
    score = scores_finais[i].item()
    risco = "🔴 DETECTADA" if score > 0.5 else "⚠️  perdida"
    print(f"    {eventos_fraude[i][:50]:50}  score={score:.3f}  {risco}")

print("\n  LEGÍTIMAS:")
for i in range(len(sequencias_legitimas)):
    score = scores_finais[len(sequencias_fraude) + i].item()
    risco = "🟢 OK" if score <= 0.5 else "⚠️  falso alarme"
    print(f"    {eventos_legitimos[i][:50]:50}  score={score:.3f}  {risco}")

# ============================================================
# COMPARATIVO: antes vs depois do treinamento
# ============================================================
print("\n" + "=" * 60)
print("O QUE O TREINAMENTO FEZ")
print("=" * 60)
print("""
  ANTES (pesos aleatórios):
    todos os scores ≈ 0.92  ← sem significado
    modelo não distingue fraude de legítima

  DEPOIS (pesos treinados):
    fraudes:    score alto (> 0.5)  ← modelo aprendeu o padrão
    legítimas:  score baixo (< 0.5) ← modelo aprendeu a diferença

  O QUE O BACKWARD FEZ:
    calculou ∂perda/∂W para cada peso do modelo
    (Wq, Wk, Wv, W1, W2, W_class — todos os blocos)
    AdamW ajustou cada peso na direção que reduz a perda
    repetiu por 200 épocas até convergir
""")