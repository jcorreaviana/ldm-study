"""
Converte o output do transformer em score de fraude via sigmoide.
Usa apenas o último token não-padding da sequência.

z     = Linear(D_MODEL, 1)(output[-1])
score = sigmoid(z)

Exporta: class FraudClassifier(nn.Module)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import importlib
from config import SEED, D_MODEL

_tra = importlib.import_module("src.07_transformer")
FraudTransformer = _tra.FraudTransformer

torch.manual_seed(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")


# ── Classe pública ─────────────────────────────────────────────────────────────
class FraudClassifier(nn.Module):
    """
    Backbone transformer + cabeça de classificação binária.
    Entrada:  x [batch, seq_len, 5], mask [batch, seq_len]
    Saída:    score [batch] ∈ [0, 1]
    """
    def __init__(self):
        super().__init__()
        self.transformer = FraudTransformer()
        self.classifier  = nn.Linear(D_MODEL, 1)

    def forward(self, x, mask=None):
        output = self.transformer(x, mask)   # [batch, seq_len, D_MODEL]

        # Índice do último evento real por sequência (evita usar padding)
        if mask is not None:
            last_idx = (mask.sum(dim=1) - 1).long()  # [batch]
        else:
            last_idx = torch.full((x.size(0),), x.size(1) - 1, dtype=torch.long)

        batch_idx  = torch.arange(x.size(0))
        last_token = output[batch_idx, last_idx]      # [batch, D_MODEL]

        logit = self.classifier(last_token).squeeze(-1)  # [batch]
        return torch.sigmoid(logit)

    def predict_with_logit(self, x, mask=None):
        """Retorna (score, logit) para inspeção didática."""
        output = self.transformer(x, mask)
        if mask is not None:
            last_idx = (mask.sum(dim=1) - 1).long()
        else:
            last_idx = torch.full((x.size(0),), x.size(1) - 1, dtype=torch.long)
        batch_idx  = torch.arange(x.size(0))
        last_token = output[batch_idx, last_idx]
        logit  = self.classifier(last_token).squeeze(-1)
        score  = torch.sigmoid(logit)
        return score, logit


# ── Main standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _tok = importlib.import_module("src.01_tokenizer")
    import numpy as np

    dados = _tok.load_and_tokenize()
    X, y, masks = dados["X"], dados["y"], dados["masks"]

    model = FraudClassifier()
    model.eval()

    # Escolher 3 exemplos: fraude fácil, normal, fraude difícil
    ids_fraude  = np.where(y == 1)[0]
    ids_normal  = np.where(y == 0)[0]

    bloco(1, "Scores para exemplos (modelo sem treino — pesos aleatórios)")
    print()
    print("  Nota: scores aleatórios — o modelo aprende em src/09_train_and_eval.py")
    print()

    casos = [
        (ids_fraude[0], "fraude"),
        (ids_normal[0], "normal"),
        (ids_fraude[1], "fraude"),
    ]

    for cid, label_str in casos:
        x_t = torch.tensor(X[cid:cid+1])
        m_t = torch.tensor(masks[cid:cid+1])
        with torch.no_grad():
            score, logit = model.predict_with_logit(x_t, m_t)
        s = score.item()
        z = logit.item()
        real_label = int(y[cid])
        pred = "FRAUDE" if s > 0.5 else "NORMAL"
        correto = "✅" if (s > 0.5) == bool(real_label) else "❌"
        print(f"  Cliente {cid:>3} (real={label_str})")
        print(f"    z (logit):    {z:+.3f}")
        print(f"    score:        {s:.3f}  →  {pred} {correto}")
        print()

    bloco(2, "Por que usar o último token")
    print()
    print("  O classificador usa output[-1] (último evento real da sequência).")
    print()
    print("  Por quê? Através das camadas de atenção, o último token acumulou")
    print("  informação de TODOS os eventos anteriores via atenção.")
    print("  É como o 'resumo' da sequência inteira.")
    print()
    print("  Alternativa seria usar o token [CLS] (BERT-style),")
    print("  mas para sequências de comprimento variável o último token funciona bem.")

    bloco(3, "Parâmetros totais")
    total = sum(p.numel() for p in model.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    backbone_params   = total - classifier_params
    print(f"\n  Backbone (transformer): {backbone_params:>7,} parâmetros")
    print(f"  Cabeça classificadora:  {classifier_params:>7,} parâmetros  (Linear {D_MODEL}→1)")
    print(f"  {'─'*40}")
    print(f"  Total:                  {total:>7,} parâmetros")
