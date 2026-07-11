"""
Treina o transformer sequencial de card testing.

Uso:
    python train.py

Se o script parecer "travado sem output": rode com `python -u train.py`
(ou `set PYTHONUNBUFFERED=1` / `export PYTHONUNBUFFERED=1` antes). Por
padrão, quando o stdout do Python não está ligado a um terminal (por
exemplo, ao redirecionar para um arquivo, rodar dentro de certas IDEs, ou
executar em background), o Python usa buffer de bloco em vez de buffer de
linha -- os prints ficam represados na memória e só aparecem quando o
buffer enche ou o processo termina. Todos os prints abaixo já usam
flush=True para não depender disso, mas o -u garante o mesmo em qualquer
ambiente.

O que faz:
1. Carrega o dataset e separa treino/val/teste por clienteID (nunca por
   transação — ver utils.split_clientes).
2. Ajusta padronização numérica + vocabulário categórico SÓ no treino.
3. Treina com BCE mascarado (ignora posições de padding) e pos_weight para
   compensar o desbalanceamento severo (~0,7% de fraude).
4. A cada época, mede AUPRC (average precision) na validação — é a métrica
   de early stopping e a métrica principal do projeto.
5. Salva o melhor checkpoint + os preprocessadores em modelo_out/, prontos
   para o evaluate.py.

Diagnóstico: cada etapa principal imprime uma linha ao ser concluída (com
timing), e o loop de treino imprime progresso a cada alguns batches dentro
da própria época -- se o script realmente travar, a última linha impressa
diz exatamente em qual etapa/batch isso aconteceu.
"""

import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from config import Config
from dataset import make_dataloader
from model import TransformerFraudDetector
from utils import load_data, split_clientes, fit_preprocessors, transform


def log(msg: str) -> None:
    """Print com flush=True -- ver nota no topo do arquivo sobre buffering."""
    print(msg, flush=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_pos_weight(df_train) -> float:
    n_pos = (df_train["isFraud"] == 1).sum()
    n_neg = (df_train["isFraud"] == 0).sum()
    return float(n_neg / max(n_pos, 1))


def masked_bce_loss(logits, labels, valid_mask, pos_weight, device):
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device), reduction="none"
    )
    loss = loss_fn(logits, labels)
    # torch.where (não multiplicação) para descartar posições de padding sem
    # risco de NaN * 0 = NaN caso algum logit de padding vire inf/nan.
    zeros = torch.zeros_like(loss)
    loss = torch.where(valid_mask.bool(), loss, zeros)
    return loss.sum() / valid_mask.float().sum().clamp(min=1.0)


@torch.no_grad()
def evaluate_auprc(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        numeric = batch["numeric"].to(device)
        tipo_idx = batch["tipo_idx"].to(device)
        merchant_idx = batch["merchant_idx"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        valid_mask = batch["valid_mask"]
        labels = batch["labels"]

        logits = model(numeric, tipo_idx, merchant_idx, attn_mask)
        probs = torch.sigmoid(logits).cpu()

        valid = valid_mask.bool()
        all_probs.append(probs[valid].numpy())
        all_labels.append(labels[valid].numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return average_precision_score(y_true, y_prob)


def main():
    t_inicio = time.time()
    log("=" * 60)
    log("INICIANDO train.py")
    log("=" * 60)
    log(f"Python: {sys.version.split()[0]} | Torch: {torch.__version__}")

    cfg = Config()
    set_seed(cfg.seed)
    cfg.out_dir.mkdir(exist_ok=True)
    log(f"[1/8] Config carregado. out_dir={cfg.out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[2/8] Device: {device}")

    # --- carga de dados ---
    t0 = time.time()
    df = load_data(cfg)
    log(f"[3/8] load_data OK -- {len(df):,} linhas, {df['clienteID'].nunique():,} clientes "
        f"({time.time() - t0:.1f}s)".replace(",", "."))

    # --- split por cliente ---
    t0 = time.time()
    train_ids, val_ids, test_ids = split_clientes(df, cfg)
    log(f"[4/8] split_clientes OK -- treino={len(train_ids)} | val={len(val_ids)} | "
        f"teste={len(test_ids)} ({time.time() - t0:.1f}s)")

    # --- preprocessadores (ajustados SÓ no treino) ---
    t0 = time.time()
    df_train_raw = df[df["clienteID"].isin(train_ids)]
    prep = fit_preprocessors(df_train_raw)
    prep.save(cfg.out_dir / "preprocessors.pkl")
    log(f"[5/8] fit_preprocessors OK -- {len(df_train_raw):,} linhas de treino | "
        f"tipos={list(prep.tipo2idx.keys())} | merchants={list(prep.merchant2idx.keys())} "
        f"({time.time() - t0:.1f}s)".replace(",", "."))

    # --- transform: aplicado no df INTEIRO (treino+val+teste), usando os
    # preprocessadores ajustados só no treino. Assinatura é transform(df, prep)
    # -- 2 argumentos posicionais, não 3 (cfg não entra aqui, só em fit_preprocessors
    # indiretamente via df_train_raw). Ver utils.py linha da definição de transform().
    t0 = time.time()
    df_t = transform(df, prep)
    log(f"[6/8] transform OK -- shape={df_t.shape} ({time.time() - t0:.1f}s)")

    # --- dataloaders ---
    t0 = time.time()
    train_loader = make_dataloader(df_t, train_ids, cfg, shuffle=True)
    val_loader = make_dataloader(df_t, val_ids, cfg, shuffle=False)
    log(f"[7/8] DataLoaders prontos -- treino={len(train_loader)} batches | "
        f"val={len(val_loader)} batches ({time.time() - t0:.1f}s)")

    pos_weight = compute_pos_weight(df_train_raw)
    log(f"pos_weight (neg/pos no treino): {pos_weight:.1f}")

    model = TransformerFraudDetector(
        n_tipo=len(prep.tipo2idx),
        n_merchant=len(prep.merchant2idx),
        n_numeric=cfg.n_numeric,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        cat_emb_dim=cfg.cat_emb_dim,
        max_len=cfg.max_seq_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[8/8] Modelo criado -- {n_params:,} parâmetros ({time.time() - t_inicio:.1f}s desde o início)"
        .replace(",", "."))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_auprc = -1.0
    epochs_sem_melhora = 0
    history = []

    n_batches_treino = len(train_loader)
    # imprime progresso a cada ~20% dos batches da época (mínimo a cada 1 batch,
    # útil sobretudo na primeira época para confirmar que o loop está vivo)
    log_a_cada = max(1, n_batches_treino // 5)

    log("")
    log("=" * 60)
    log(f"INICIANDO LOOP DE TREINO ({cfg.epochs} épocas no máximo, "
        f"{n_batches_treino} batches/época, CPU pode ser lento -- "
        f"progresso a cada {log_a_cada} batches)")
    log("=" * 60)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            numeric = batch["numeric"].to(device)
            tipo_idx = batch["tipo_idx"].to(device)
            merchant_idx = batch["merchant_idx"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(numeric, tipo_idx, merchant_idx, attn_mask)
            loss = masked_bce_loss(logits, labels, valid_mask, pos_weight, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % log_a_cada == 0 or batch_idx == n_batches_treino:
                log(f"  época {epoch:03d} | batch {batch_idx}/{n_batches_treino} | "
                    f"loss médio até aqui={total_loss / n_batches:.4f} | "
                    f"{time.time() - t0:.1f}s decorridos")

        train_loss = total_loss / max(n_batches, 1)

        log(f"  época {epoch:03d} | treino concluído, iniciando validação...")
        t_val = time.time()
        val_auprc = evaluate_auprc(model, val_loader, device)
        dt = time.time() - t0

        log(
            f"época {epoch:03d} | loss treino={train_loss:.4f} | "
            f"AUPRC validação={val_auprc:.4f} | validação={time.time() - t_val:.1f}s | "
            f"época inteira={dt:.1f}s"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auprc": val_auprc})

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            epochs_sem_melhora = 0
            torch.save(model.state_dict(), cfg.out_dir / "model.pt")
            with open(cfg.out_dir / "model_config.json", "w") as f:
                json.dump(
                    {
                        "n_tipo": len(prep.tipo2idx),
                        "n_merchant": len(prep.merchant2idx),
                        "n_numeric": cfg.n_numeric,
                        "d_model": cfg.d_model,
                        "n_heads": cfg.n_heads,
                        "n_layers": cfg.n_layers,
                        "dim_feedforward": cfg.dim_feedforward,
                        "dropout": cfg.dropout,
                        "cat_emb_dim": cfg.cat_emb_dim,
                        "max_len": cfg.max_seq_len,
                        "window_size": cfg.window_size,
                        "best_val_auprc": best_auprc,
                        "best_epoch": epoch,
                    },
                    f,
                    indent=2,
                )
            log(f"  -> novo melhor checkpoint salvo (AUPRC={best_auprc:.4f})")
        else:
            epochs_sem_melhora += 1
            log(f"  -> sem melhora ({epochs_sem_melhora}/{cfg.patience})")
            if epochs_sem_melhora >= cfg.patience:
                log(f"Early stopping — sem melhora por {cfg.patience} épocas.")
                break

    with open(cfg.out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ids de teste ficam salvos para o evaluate.py usar exatamente o mesmo split
    with open(cfg.out_dir / "test_client_ids.json", "w") as f:
        json.dump(list(test_ids), f)

    log(f"\nMelhor AUPRC de validação: {best_auprc:.4f}")
    log(f"Artefatos salvos em: {cfg.out_dir}")
    log(f"Tempo total: {time.time() - t_inicio:.1f}s")


if __name__ == "__main__":
    main()
