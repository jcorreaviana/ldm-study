"""
Adiciona informação de posição temporal ao embedding de cada evento.
Usa positional encoding sinusoidal (Vaswani et al., 2017).

PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Exporta: class PositionalEncoding(nn.Module)
"""
import sys, os, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from config import SEED, D_MODEL, SEQ_LEN

torch.manual_seed(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")

def imprimir_pe(pos, vetor):
    primeiros = vetor[:8].tolist()
    fmt = "[ " + ", ".join(f"{v:6.3f}" for v in primeiros) + ", ... ]"
    print(f"  pos={pos}:  {fmt}")


# ── Classe pública ─────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """
    Soma um vetor sinusoidal fixo (não aprendido) a cada posição da sequência.
    Entrada:  [batch, seq_len, D_MODEL]
    Saída:    [batch, seq_len, D_MODEL]
    """
    def __init__(self, d_model=D_MODEL, max_len=SEQ_LEN):
        super().__init__()
        # Calcula a tabela PE uma vez e registra como buffer (não é parâmetro)
        pe = torch.zeros(max_len, d_model)
        posicao = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(posicao * div_term)
        pe[:, 1::2] = torch.cos(posicao * div_term)
        # [1, max_len, d_model] — broadcast sobre o batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


# ── Main standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pe_layer = PositionalEncoding()

    bloco(1, "Tabela de positional encodings (primeiras 8 dims)")
    print()
    for pos in [0, 1, 5, 9]:
        imprimir_pe(pos, pe_layer.pe[0, pos])

    bloco(2, "Diferença entre posição 0 e posição 6")
    pe_0 = pe_layer.pe[0, 0]
    pe_6 = pe_layer.pe[0, 6]
    dist = torch.norm(pe_0 - pe_6).item()
    print(f"\n  Distância L2 entre PE(pos=0) e PE(pos=6): {dist:.3f}")
    print("  Posições diferentes → vetores diferentes → o modelo distingue a ordem.")

    bloco(3, "Embedding ANTES e DEPOIS do positional encoding")
    # Simular um token qualquer
    torch.manual_seed(SEED)
    token_ficticio = torch.randn(1, SEQ_LEN, D_MODEL)

    print(f"\n  Evento na posição 6 — primeiras 8 dims:")
    antes = token_ficticio[0, 6]
    print("  ANTES  pos enc: [ " + ", ".join(f"{v:6.3f}" for v in antes[:8].tolist()) + ", ... ]")

    com_pe = pe_layer(token_ficticio)
    depois = com_pe[0, 6]
    print("  DEPOIS pos enc: [ " + ", ".join(f"{v:6.3f}" for v in depois[:8].tolist()) + ", ... ]")

    delta = depois - antes
    print("  Delta (PE):     [ " + ", ".join(f"{v:6.3f}" for v in delta[:8].tolist()) + ", ... ]")

    bloco(4, "Por que positional encoding importa para fraude")
    print()
    print("  Sem positional encoding o transformer trataria:")
    print('    "troca_senha no evento 2"  ==  "troca_senha no evento 9"')
    print()
    print("  Com positional encoding:")
    print("    evento 9 (fim da sequência) recebe mais atenção do classificador")
    print("    porque é o evento mais recente — contexto crucial para fraude.")
    print()
    print("  Um evento suspeito no FINAL da sequência é muito mais relevante")
    print("  do que o mesmo evento no INÍCIO.")
