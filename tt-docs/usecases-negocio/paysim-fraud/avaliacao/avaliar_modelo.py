"""
Avalia o checkpoint treinado: curva precision-recall, matriz de confusao e
resultado_final.md com as metricas reais (nao estimadas).

So funciona depois que voce rodou modelo/transformer_sequencial.py e existe
um checkpoint_melhor_modelo.pt de verdade - este script nao inventa metricas.

Avalia SO no split "teste" do event_stream.npz (por conta, gerado em
montar_event_stream.py) - nunca visto no treino nem na validacao. Antes
avaliava o arquivo inteiro misturado (treino+val+teste); corrigido junto
com o fix de vazamento em modelo/transformer_sequencial.py.

Rode: python avaliar_modelo.py --data ../../paysim_data/event_stream.npz --checkpoint ../modelo/checkpoint_melhor_modelo.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
)
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent / "modelo"))
from transformer_sequencial import (
    EventStreamDataset,
    TransformerSequencial,
    carregar_event_stream,
    montar_batch,
)


@torch.no_grad()
def prever(modelo, dl, device):
    todos_probs, todos_labels = [], []
    for tipo, numericas, delta_step, mascara, label in dl:
        tipo, numericas, delta_step = tipo.to(device), numericas.to(device), delta_step.to(device)
        mascara = mascara.to(device)
        logit = modelo(tipo, numericas, delta_step, mascara)
        todos_probs.append(torch.sigmoid(logit).cpu().numpy())
        todos_labels.append(label.numpy())
    return np.concatenate(todos_probs), np.concatenate(todos_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--janela", type=int, default=13)
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="definido com o cliente - ver FASE 7 do script-desafio.md")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)

    modelo = TransformerSequencial(janela=args.janela).to(device)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()

    # avaliacao final: usa SO o split "teste" (nunca visto no treino nem na
    # validacao) - consistente com o split por conta feito em
    # montar_event_stream.py. Ver EventStreamDataset/carregar_event_stream
    # em modelo/transformer_sequencial.py.
    X, y, seq_lens, split = carregar_event_stream(args.data)
    idx_teste = np.where(split == "teste")[0]
    if len(idx_teste) == 0:
        raise ValueError(
            "split 'teste' vazio no event_stream.npz - confira se "
            "montar_event_stream.py rodou corretamente."
        )
    dataset = EventStreamDataset(X[idx_teste], y[idx_teste], seq_lens[idx_teste])
    print(f"avaliando no split de teste: {len(dataset):,} exemplos")
    dl = DataLoader(dataset, batch_size=256,
                     collate_fn=lambda b: montar_batch(b, args.janela))

    probs, labels = prever(modelo, dl, device)

    auprc = average_precision_score(labels, probs)
    precisao, recall, thresholds = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precisao, color="#378ADD")
    ax.set_xlabel("recall")
    ax.set_ylabel("precisao")
    ax.set_title(f"Curva PR (AUPRC = {auprc:.4f})")
    fig.savefig(Path(__file__).parent / "curva_pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    preds_bin = (probs >= args.threshold).astype(int)
    cm = confusion_matrix(labels, preds_bin)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Previsto legitima", "Previsto fraude"])
    ax.set_yticks([0, 1], ["Real legitima", "Real fraude"])
    ax.set_title(f"Matriz de confusao (threshold={args.threshold})")
    fig.savefig(Path(__file__).parent / "matriz_confusao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    recall_final = recall_score(labels, preds_bin)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    decisoes = """## Decisões de configuração confirmadas

Baseado no censo completo de `eda/analisar_namedest.py` (509.565 contas
`nameDest` únicas, TRANSFER/CASH_OUT):

| Decisão | Valor | Justificativa |
|---|---|---|
| Entidade da sequência | `nameDest` | `nameOrig` não se repete (ver EDA_paysim.md seção 5-7) |
| Label | por posição/transação | 5.062 contas (0,99%) são mistas (fraude + legítima) — label por conta classificaria errado as transações legítimas dessas contas |
| Janela de contexto | 13 eventos | p90 do censo = 13 (cobre 90% das contas sem truncar); mediana real é só 3 (curta demais); VRAM da RTX 5060 (~8,5GB) limita janelas maiores |
| Corte de vazamento | contexto = só eventos anteriores à posição rotulada | necessário para a meta "detectar antes da fraude principal" |
| Sinal de volume | fraco isoladamente | contas com fraude têm média 5,65 transações vs 5,43 nas legítimas — diferença desprezível; o transformer precisa aprender o padrão da sequência, não a contagem |
| Split treino/val/teste | por conta (`nameDest`), não por exemplo | corrigido vazamento: `random_split()` por exemplo virou split pelo campo `split` do `.npz` (por conta) |
"""

    resultado = f"""# Resultado final — transformer sequencial (PaySim)

{decisoes}
## Métricas (geradas por avaliar_modelo.py em {args.checkpoint})

Checkpoint avaliado: `{args.checkpoint}`
Threshold usado: {args.threshold} (definir com o cliente antes de finalizar)

| Métrica | Valor |
|---|---|
| AUPRC (split de teste, {len(idx_teste):,} exemplos) | {auprc:.4f} |
| Recall | {recall_final:.4f} |
| FPR (falso positivo) | {fpr:.4f} |
| TP / FP / FN / TN | {tp} / {fp} / {fn} / {tn} |

Meta técnica do script-desafio.md: AUPRC > 0.80 (comparar com baseline
tabular anterior de 0.705).

Ver `curva_pr.png` e `matriz_confusao.png` nesta pasta.
"""
    with open(Path(__file__).parent / "resultado_final.md", "w") as f:
        f.write(resultado)

    print(resultado)


if __name__ == "__main__":
    main()
