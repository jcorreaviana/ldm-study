"""
Positional encoding para o event stream do PaySim.

Diferente de um transformer de texto (onde a posicao e so o indice do
token), aqui a ordem importa mas o ESPACAMENTO no tempo tambem: duas
transacoes que chegam na mesma conta com 1 step de diferenca nao sao
equivalentes a duas que chegam com 300 steps de diferenca. Por isso, alem
da codificacao senoidal classica por posicao no vetor, este modulo tambem
aceita o `delta_step` (tempo desde o evento anterior) como sinal continuo.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Codificacao senoidal classica (Vaswani et al.), por indice na sequencia."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        posicao = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(posicao * div_term)
        pe[:, 1::2] = torch.cos(posicao * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TimeAwarePositionalEncoding(nn.Module):
    """
    Combina a codificacao senoidal por indice com uma projecao aprendida do
    `delta_step` (tempo real desde o evento anterior daquela conta destino).
    Use esta variante quando o espacamento temporal entre eventos for
    irregular (o caso do PaySim - contas nao recebem transacoes em intervalos
    fixos).
    """

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.indice_pe = PositionalEncoding(d_model, max_len)
        self.projecao_delta = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor, delta_step: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) | delta_step: (batch, seq_len, 1)
        x = self.indice_pe(x)
        return x + self.projecao_delta(delta_step)
