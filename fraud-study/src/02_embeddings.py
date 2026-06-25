"""
Transforma índices numéricos em vetores densos de D_MODEL dimensões.
Campos categóricos → nn.Embedding (tabela de lookup).
Campos numéricos   → nn.Linear   (projeção linear).

Exporta: class FieldEncoders(nn.Module)
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from config import SEED, D_MODEL, TIPOS, PAISES, DEVICES

torch.manual_seed(SEED)

# ── Helpers ────────────────────────────────────────────────────────────────────
def bloco(n, titulo):
    print(f"\n{'═'*60}")
    print(f"  BLOCO {n} — {titulo}")
    print(f"{'═'*60}")

def ok(msg, start=None):
    elapsed = f"  ({time.time()-start:.1f}s)" if start else ""
    print(f"✅ {msg}{elapsed}")

def imprimir_vetor(nome, vetor, nota=""):
    primeiros = vetor[:6].tolist()
    fmt = "[ " + ", ".join(f"{v:6.2f}" for v in primeiros) + ", ... ]"
    nota_str = f"  ← {nota}" if nota else ""
    print(f"  {nome:<8} [{D_MODEL} dims]: {fmt}{nota_str}")


# ── Classe pública ─────────────────────────────────────────────────────────────
class FieldEncoders(nn.Module):
    """
    Um encoder por campo de entrada.
    Categóricos usam nn.Embedding (lookup table aprendida).
    Numéricos usam nn.Linear (projeção escalar → vetor).
    """
    def __init__(self):
        super().__init__()
        self.emb_tipo   = nn.Embedding(len(TIPOS),   D_MODEL)
        self.emb_pais   = nn.Embedding(len(PAISES),  D_MODEL)
        self.emb_device = nn.Embedding(len(DEVICES), D_MODEL)
        self.lin_valor  = nn.Linear(1, D_MODEL)
        self.lin_hora   = nn.Linear(1, D_MODEL)

    def encode_evento(self, tipo_idx, pais_idx, device_idx, valor_norm, hora_norm):
        """
        Codifica um único evento. Retorna dict com vetor por campo.
        Entradas são tensores escalares ou de shape [1].
        """
        tipo_t   = torch.tensor([tipo_idx],   dtype=torch.long)
        pais_t   = torch.tensor([pais_idx],   dtype=torch.long)
        device_t = torch.tensor([device_idx], dtype=torch.long)
        valor_t  = torch.tensor([[valor_norm]], dtype=torch.float32)
        hora_t   = torch.tensor([[hora_norm]],  dtype=torch.float32)

        return {
            "tipo":   self.emb_tipo(tipo_t).squeeze(0),
            "pais":   self.emb_pais(pais_t).squeeze(0),
            "device": self.emb_device(device_t).squeeze(0),
            "valor":  self.lin_valor(valor_t).squeeze(0),
            "hora":   self.lin_hora(hora_t).squeeze(0),
        }

    def forward(self, x):
        """
        x: [batch, seq_len, 5]  — (tipo, pais, device, valor_norm, hora_norm)
        Retorna dict de tensores [batch, seq_len, D_MODEL], um por campo.
        """
        tipo_idx   = x[:, :, 0].long()
        pais_idx   = x[:, :, 1].long()
        device_idx = x[:, :, 2].long()
        valor      = x[:, :, 3:4]
        hora       = x[:, :, 4:5]

        return {
            "tipo":   self.emb_tipo(tipo_idx),
            "pais":   self.emb_pais(pais_idx),
            "device": self.emb_device(device_idx),
            "valor":  self.lin_valor(valor),
            "hora":   self.lin_hora(hora),
        }


# ── Main standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    encoders = FieldEncoders()
    encoders.eval()

    # Evento NORMAL: login, BR, celular_habitual, sem valor, 14h
    bloco(1, "Embedding de evento NORMAL (login BR celular 14h)")
    enc_normal = encoders.encode_evento(
        tipo_idx=0,   # login
        pais_idx=0,   # BR
        device_idx=0, # celular_habitual
        valor_norm=0.0,
        hora_norm=14/23,
    )
    for campo, vetor in enc_normal.items():
        imprimir_vetor(campo, vetor.detach())

    # Evento SUSPEITO: troca_senha, JP, dispositivo_desconhecido, 3h
    bloco(2, "Embedding de evento SUSPEITO (troca_senha JP dispositivo_desc 3h)")
    enc_suspeito = encoders.encode_evento(
        tipo_idx=3,   # troca_senha
        pais_idx=3,   # JP
        device_idx=3, # dispositivo_desconhecido
        valor_norm=0.0,
        hora_norm=3/23,
    )
    for campo, vetor in enc_suspeito.items():
        imprimir_vetor(campo, vetor.detach())

    # Distância entre os dois eventos
    bloco(3, "Distância entre os dois eventos no espaço vetorial")
    # Concatenar todos os campos em um vetor único por evento
    vec_normal   = torch.cat(list(enc_normal.values()))
    vec_suspeito = torch.cat(list(enc_suspeito.values()))
    dist = torch.norm(vec_normal - vec_suspeito).item()

    print(f"\n  Vetor normal   : {D_MODEL*5} dims (5 campos × {D_MODEL})")
    print(f"  Vetor suspeito : {D_MODEL*5} dims (5 campos × {D_MODEL})")
    print(f"\n  Distância L2 entre os dois eventos: {dist:.2f}")
    print()
    print("  Tokens diferentes ficam distantes no espaço vetorial.")
    print("  Após o treino, eventos suspeitos formam clusters separados.")

    bloco(4, "Parâmetros aprendíveis")
    total = sum(p.numel() for p in encoders.parameters())
    print(f"\n  emb_tipo   : {encoders.emb_tipo.weight.numel():>6} parâmetros")
    print(f"  emb_pais   : {encoders.emb_pais.weight.numel():>6} parâmetros")
    print(f"  emb_device : {encoders.emb_device.weight.numel():>6} parâmetros")
    lin_params = sum(p.numel() for p in encoders.lin_valor.parameters()) + \
                 sum(p.numel() for p in encoders.lin_hora.parameters())
    print(f"  lin_valor + lin_hora: {lin_params:>6} parâmetros")
    print(f"  {'─'*35}")
    print(f"  Total FieldEncoders : {total:>6} parâmetros")
