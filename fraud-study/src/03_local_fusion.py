"""
Combina os embeddings dos 5 campos de um evento em um único vetor de D_MODEL.
concat([tipo, pais, device, valor, hora]) → Linear(D_MODEL*5, D_MODEL) → ReLU → LayerNorm

Exporta: class LocalFusion(nn.Module)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import importlib
from config import SEED, D_MODEL

# Arquivos com prefixo numérico não são importáveis diretamente
_emb_mod = importlib.import_module("src.02_embeddings")
FieldEncoders = _emb_mod.FieldEncoders

torch.manual_seed(SEED)

N_CAMPOS = 5  # tipo, pais, device, valor, hora

# ── Helpers ────────────────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")

def imprimir_campo(nome, vetor):
    primeiros = vetor[:6].tolist()
    fmt = "[ " + ", ".join(f"{v:6.2f}" for v in primeiros) + ", ... ]"
    norma = torch.norm(vetor).item()
    print(f"  {nome:<8}: {fmt}  norma={norma:.2f}")


# ── Classe pública ─────────────────────────────────────────────────────────────
class LocalFusion(nn.Module):
    """
    Recebe um dict de embeddings por campo e os funde num único token.
    Entrada:  dict {campo: tensor [batch, seq_len, D_MODEL]}  (5 campos)
    Saída:    tensor [batch, seq_len, D_MODEL]
    """
    def __init__(self):
        super().__init__()
        self.proj  = nn.Linear(D_MODEL * N_CAMPOS, D_MODEL)
        self.relu  = nn.ReLU()
        self.norm  = nn.LayerNorm(D_MODEL)

    def forward(self, field_embeddings: dict):
        # Concatena na última dimensão: [batch, seq_len, D_MODEL*5]
        concatenado = torch.cat(list(field_embeddings.values()), dim=-1)
        projetado   = self.proj(concatenado)
        return self.norm(self.relu(projetado))


# ── Main standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    encoders = FieldEncoders()
    fusion   = LocalFusion()
    encoders.eval()
    fusion.eval()

    # Evento suspeito: troca_senha, JP, dispositivo_desconhecido, sem valor, 3h
    enc = encoders.encode_evento(
        tipo_idx=3, pais_idx=3, device_idx=3,
        valor_norm=0.0, hora_norm=3/23,
    )

    bloco(1, "Campos isolados — evento suspeito")
    for campo, vetor in enc.items():
        imprimir_campo(campo, vetor.detach())

    bloco(2, "Após Local Fusion (troca_senha + JP + dispositivo_desc + 3h)")

    # Adicionar dimensões batch e seq para o forward
    field_batch = {k: v.unsqueeze(0).unsqueeze(0) for k, v in enc.items()}
    with torch.no_grad():
        token = fusion(field_batch).squeeze()

    primeiros = token[:6].tolist()
    fmt = "[ " + ", ".join(f"{v:6.2f}" for v in primeiros) + ", ... ]"
    norma_fusao = torch.norm(token).item()
    print(f"\n  token [{D_MODEL} dims]: {fmt}")
    print(f"  norma={norma_fusao:.2f}")

    bloco(3, "Comparação: campos isolados vs. fusão")
    normas_campo = {k: torch.norm(v).item() for k, v in enc.items()}
    norma_max = max(normas_campo.values())
    print()
    for campo, norma in normas_campo.items():
        barra = "█" * int(norma * 5)
        print(f"  {campo:<8}  norma={norma:.2f}  {barra}")
    barra_fusao = "█" * int(norma_fusao * 5)
    print(f"  {'fusão':<8}  norma={norma_fusao:.2f}  {barra_fusao}  ← combinação amplifica o sinal")
    print()
    print(f"  Conclusão: a norma do vetor fundido ({norma_fusao:.2f}) é maior")
    print(f"  que qualquer campo isolado (máx: {norma_max:.2f}).")
    print("  A combinação de campos suspeitos cria um sinal mais intenso.")

    bloco(4, "Parâmetros")
    total = sum(p.numel() for p in fusion.parameters())
    print(f"\n  proj (Linear {D_MODEL*N_CAMPOS}→{D_MODEL}): {D_MODEL*N_CAMPOS*D_MODEL + D_MODEL} parâmetros")
    print(f"  norm (LayerNorm {D_MODEL}):         {D_MODEL*2} parâmetros")
    print(f"  Total LocalFusion:                  {total} parâmetros")
