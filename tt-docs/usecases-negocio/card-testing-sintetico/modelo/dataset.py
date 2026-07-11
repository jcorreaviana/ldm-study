"""
Dataset por cliente + colagem (collate) em lote com padding dinâmico e
máscara de atenção "banded causal": cada transação (posição i) só pode
atender a si mesma e às até (window_size - 1) transações imediatamente
anteriores. Isso implementa diretamente o requisito de negócio:

    "cada transação avaliada com contexto das últimas N transações"

sem vazar informação de transações futuras (que ainda não aconteceram no
momento em que a transação i precisaria ser escorada em produção).

Label de treino: usamos 'label_preventivo' (não 'isFraud' puro) como alvo.
'isFraud' original continua disponível em cada item como 'isfraud_real',
para quem precisar do rótulo verdadeiro (ex.: métricas de avaliação que
comparam contra a fraude de fato, não contra o rótulo propagado). Ver
utils.propagate_label_preventivo para a lógica de propagação e a
motivação (AUPRC=1.0 mas recall nas micro-transações=0% ao treinar contra
isFraud puro).
"""

from functools import partial
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from utils import FEATURES_NUMERICAS, PAD_IDX, propagate_label_preventivo


class ClientSequenceDataset(Dataset):
    """Um item = a sequência completa (até max_seq_len) de um cliente."""

    def __init__(self, df: pd.DataFrame, cliente_ids, max_seq_len: int):
        self.max_seq_len = max_seq_len
        df = df[df["clienteID"].isin(set(cliente_ids))]
        # agrupa uma vez só, na criação do dataset (não a cada __getitem__)
        self.groups = {
            cid: g.sort_values("timestamp").reset_index(drop=True)
            for cid, g in df.groupby("clienteID")
        }
        self.ids: List[str] = list(self.groups.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        cid = self.ids[idx]
        g = self.groups[cid]

        if len(g) > self.max_seq_len:
            # mantém as transações mais recentes — é onde o golpe, se houver, está
            g = g.iloc[-self.max_seq_len:].reset_index(drop=True)

        numeric_cols = [f"{c}_norm" for c in FEATURES_NUMERICAS]
        numeric = torch.tensor(g[numeric_cols].values, dtype=torch.float32)
        tipo_idx = torch.tensor(g["tipo_idx"].values, dtype=torch.long)
        merchant_idx = torch.tensor(g["merchant_idx"].values, dtype=torch.long)
        # alvo de treino = label_preventivo (propagado), não isFraud puro
        labels = torch.tensor(g["label_preventivo"].values, dtype=torch.float32)
        # isFraud original preservado à parte, sem ser tocado -- útil para
        # quem precisar avaliar contra o rótulo real de fraude
        isfraud_real = torch.tensor(g["isFraud"].values, dtype=torch.float32)
        valor_raw = torch.tensor(g["valor"].values, dtype=torch.float32)
        # timestamps guardados como string p/ conseguir religar previsão -> transação original na avaliação
        timestamps = g["timestamp"].astype(str).tolist()

        return {
            "clienteID": cid,
            "numeric": numeric,
            "tipo_idx": tipo_idx,
            "merchant_idx": merchant_idx,
            "labels": labels,
            "isfraud_real": isfraud_real,
            "valor_raw": valor_raw,
            "timestamps": timestamps,
            "length": len(g),
        }


def build_banded_causal_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Retorna máscara aditiva (seq_len, seq_len) para nn.TransformerEncoder:
    0.0 onde a atenção é permitida, -inf onde é bloqueada.

    Posição i pode atender a j se: j <= i (causal, sem ver o futuro) E
    i - j < window_size (só as últimas `window_size` transações, incluindo a própria).
    """
    idx = torch.arange(seq_len)
    i = idx.view(-1, 1)
    j = idx.view(1, -1)
    permitido = (j <= i) & (i - j < window_size)
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(~permitido, float("-inf"))
    return mask


def collate_fn(batch: List[Dict], window_size: int) -> Dict:
    lengths = [b["length"] for b in batch]
    max_len = max(lengths)
    B = len(batch)

    numeric = torch.zeros(B, max_len, len(FEATURES_NUMERICAS))
    tipo_idx = torch.full((B, max_len), PAD_IDX, dtype=torch.long)
    merchant_idx = torch.full((B, max_len), PAD_IDX, dtype=torch.long)
    labels = torch.zeros(B, max_len)
    isfraud_real = torch.zeros(B, max_len)
    valor_raw = torch.zeros(B, max_len)
    key_padding_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = posição de padding
    cliente_ids, timestamps_batch = [], []

    for i, b in enumerate(batch):
        L = b["length"]
        numeric[i, :L] = b["numeric"]
        tipo_idx[i, :L] = b["tipo_idx"]
        merchant_idx[i, :L] = b["merchant_idx"]
        labels[i, :L] = b["labels"]
        isfraud_real[i, :L] = b["isfraud_real"]
        valor_raw[i, :L] = b["valor_raw"]
        key_padding_mask[i, :L] = False
        cliente_ids.append(b["clienteID"])
        timestamps_batch.append(b["timestamps"])

    attn_mask = build_banded_causal_mask(max_len, window_size)
    valid_mask = ~key_padding_mask

    return {
        "numeric": numeric,
        "tipo_idx": tipo_idx,
        "merchant_idx": merchant_idx,
        "labels": labels,
        "isfraud_real": isfraud_real,
        "valor_raw": valor_raw,
        "attn_mask": attn_mask,
        "key_padding_mask": key_padding_mask,
        "valid_mask": valid_mask,
        "clienteID": cliente_ids,
        "timestamps": timestamps_batch,
    }


def make_dataloader(df, cliente_ids, cfg, shuffle: bool) -> DataLoader:
    # garante que 'label_preventivo' exista antes de montar o Dataset --
    # assim quem chama make_dataloader (train.py, evaluate.py) não precisa
    # ser alterado: o dataset.py se vira sozinho para calcular a coluna nova.
    if "label_preventivo" not in df.columns:
        df = propagate_label_preventivo(df, cfg)

    dataset = ClientSequenceDataset(df, cliente_ids, cfg.max_seq_len)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, window_size=cfg.window_size),
    )
