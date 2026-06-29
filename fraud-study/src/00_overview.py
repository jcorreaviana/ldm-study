"""
Mapa do projeto — fluxo completo do transformer em pseudocódigo comentado.
Não implementa nada: serve como orientação antes de mergulhar nos blocos.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import D_MODEL, N_HEADS, N_LAYERS, D_FF, SEQ_LEN

DIAGRAMA = f"""
╔══════════════════════════════════════════════════════════════╗
║         TRANSFORMER FRAUD DETECTION — FLUXO COMPLETO        ║
╚══════════════════════════════════════════════════════════════╝

DADO BRUTO (1 evento de 1 cliente)
  cliente_id=5, evento_num=6, tipo="troca_senha",
  pais="JP", device="dispositivo_desconhecido",
  valor=0.0, hora=3
       ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[BLOCO 1] FIELD ENCODERS  →  src/02_embeddings.py
  Cada campo vira um vetor de {D_MODEL} dimensões independentemente.

  tipo   → nn.Embedding(7, {D_MODEL})  → vetor [{D_MODEL}]   ← tabela de lookup
  pais   → nn.Embedding(6, {D_MODEL})  → vetor [{D_MODEL}]   ← tabela de lookup
  device → nn.Embedding(4, {D_MODEL})  → vetor [{D_MODEL}]   ← tabela de lookup
  valor  → nn.Linear(1, {D_MODEL})     → vetor [{D_MODEL}]   ← projeção linear
  hora   → nn.Linear(1, {D_MODEL})     → vetor [{D_MODEL}]   ← projeção linear
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ↓
[BLOCO 2] LOCAL FUSION  →  src/03_local_fusion.py
  Os 5 vetores de campo são concatenados e fundidos num único token.

  concat([tipo, pais, device, valor, hora]) → [{D_MODEL*5}]
  nn.Linear({D_MODEL*5}, {D_MODEL})         → token_embedding [{D_MODEL}]
  ReLU + LayerNorm
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ↓
[BLOCO 3] POSITIONAL ENCODING  →  src/04_positional.py
  O token precisa saber sua posição na sequência de {SEQ_LEN} eventos.
  Sem isso o transformer trataria evento 2 igual ao evento 9.

  token_embedding + PE(6) → [{D_MODEL}]
  PE usa senos/cossenos: sin(pos / 10000^(2i/d))  (Vaswani et al., 2017)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ↓  (repetido para todos os {SEQ_LEN} eventos → matrix [{SEQ_LEN} × {D_MODEL}])
[BLOCO 4] MULTI-HEAD ATTENTION  →  src/05_attention.py + src/06_multihead.py
  Cada evento "olha" para todos os outros e decide o quanto cada um importa.

  {N_HEADS} cabeças × atenção independente (d_head = {D_MODEL//N_HEADS})
    Q = Linear({D_MODEL}, {D_MODEL//N_HEADS})(X)   ← o que estou procurando?
    K = Linear({D_MODEL}, {D_MODEL//N_HEADS})(X)   ← o que cada evento oferece?
    V = Linear({D_MODEL}, {D_MODEL//N_HEADS})(X)   ← o que de fato extraio?
    scores = softmax(Q·Kᵀ / √{D_MODEL//N_HEADS})
    head_i = scores · V
  concat(head_1...head_{N_HEADS}) → Linear({D_MODEL}, {D_MODEL}) → [{D_MODEL}]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ↓
[BLOCO 5] FEED-FORWARD + NORM  (dentro de src/07_transformer.py)
  Transformação não-linear por posição (cada evento independentemente).

  LayerNorm(X + Attention(X))         ← residual connection
  Linear({D_MODEL}→{D_FF}) + ReLU + Linear({D_FF}→{D_MODEL})
  LayerNorm(X + FF(X))                ← residual connection

  Repetido {N_LAYERS}× (N_LAYERS camadas empilhadas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ↓
[BLOCO 6] CLASSIFIER  →  src/08_classifier.py
  Usa apenas o último token (evento mais recente) para classificar.

  output[-1]  →  [{D_MODEL}]
  Linear({D_MODEL}, 1)  →  z  (logit)
  sigmoid(z)            →  score ∈ [0, 1]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       ↓
SCORE = 0.94  →  FRAUDE DETECTADA 🚨

Comparação final em src/09_train_and_eval.py:
  Transformer (vê a SEQUÊNCIA) vs XGBoost (vê AGREGAÇÕES)
  O transformer acerta casos onde a ORDEM dos eventos importa.
"""

if __name__ == "__main__":
    print(DIAGRAMA)
    print("─" * 62)
    print("  Execute cada módulo na ordem para ver cada bloco em ação:")
    print()
    print("  python data/generate_data.py")
    print("  python src/01_tokenizer.py")
    print("  python src/02_embeddings.py")
    print("  python src/03_local_fusion.py")
    print("  python src/04_positional.py")
    print("  python src/05_attention.py")
    print("  python src/06_multihead.py")
    print("  python src/07_transformer.py")
    print("  python src/08_classifier.py")
    print("  python src/09_train_and_eval.py")
    print()
