"""
Transformer sequencial para deteccao de conta-destino "laranja" no PaySim.

Diferenca central para o projeto anterior (credit card, tabular): aqui a
atencao e ENTRE TRANSACOES da mesma conta de destino (event stream), nao
entre features de uma unica transacao. Ver EDA_paysim.md secao 7 para a
justificativa de por que a entidade e nameDest e nao um clienteID (que nao
existe neste dataset).

Requisitos (ja no script-desafio.md):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  pip install pandas scikit-learn matplotlib seaborn wandb pynvml

Este script SO define a arquitetura e o loop de treino - rodar de verdade
(e gerar checkpoint_melhor_modelo.pt) requer os tensores produzidos por
../preprocessing/montar_event_stream.py e a GPU local (RTX 5060/5090).
Nenhum checkpoint esta incluido neste repositorio.

Rode: python transformer_sequencial.py --data ../../paysim_data/event_stream.npz --janela 13

Janela=13 e a mesma usada em montar_event_stream.py - ver docstring la e
avaliacao/resultado_final.md para a justificativa (censo completo de
analisar_namedest.py: p90=13, mediana=3, limite de VRAM da RTX 5060).

Correcoes feitas apos o primeiro treino reportar perda=nan e auprc=0.0012
na epoca 1:
  - validacao de NaN/Inf em X/y no __init__ do Dataset (EventStreamDataset)
  - perda (BCEWithLogitsLoss com pos_weight alto, por causa do
    desbalanceamento) calculada FORA do autocast, em fp32 - a causa mais
    provavel do perda=nan e overflow em fp16 quando pos_weight e grande
  - grad clipping (clip_grad_norm_, max_norm=1.0) apos scaler_amp.unscale_
  - batches com perda nao finita sao pulados (sem atualizar pesos) em vez
    de deixar nan contaminar o resto do treino
  - torch.cuda.amp.autocast/GradScaler (deprecated no PyTorch 2.x) trocados
    por torch.amp.autocast('cuda', ...) / torch.amp.GradScaler('cuda', ...)
  - print(f"usando device: {device}") logo no inicio de treinar(), para
    confirmar que a GPU esta sendo usada (141s/epoca seria normal em CPU,
    nao na RTX 5060)

GPU ativa e dados sem NaN/Inf confirmados, mas perda ainda nao-finita em
varios batches da epoca 1 - causa: pos_weight bruto (~836x, pela taxa de
fraude de 0.12%) estourava mesmo em fp32. Correcoes adicionais:
  - pos_weight capado em 50.0 por padrao (--pos-weight-max)
  - print do pos_weight bruto e do valor capado no inicio do treino
  - alternativa --loss focal: troca BCEWithLogitsLoss por FocalLoss
    (--focal-alpha, --focal-gamma), que nao depende de um fator
    multiplicativo enorme para lidar com desbalanceamento extremo

FocalLoss tambem retornou perda nao-finita, com dados confirmados sem
NaN/Inf - ou seja, o problema e ANTERIOR a loss (forward pass ou
inicializacao), nao a funcao de perda em si. Diagnosticos adicionados
(nenhuma mudanca de comportamento de treino, so instrumentacao):
  - EventStreamDataset imprime dtype de X e min/max/media/std por canal
    (tipo_idx, amount, delta_step, oldbalanceDest, newbalanceDest) - sem
    NaN/Inf ainda pode haver valores na casa dos milhoes (saldo do PaySim),
    que e a suspeita principal: montar_event_stream.py nao normaliza essas
    features antes de entrarem no nn.Linear/TransformerEncoder
  - diagnosticar_primeiro_batch(): roda o forward passo a passo (sem grad)
    so no primeiro batch, imprimindo dtype, modelo.embed_tipo.weight, e
    isfinite()+min/max/media em cada estagio (emb_tipo, emb_num, x antes e
    depois do positional encoding, saida do encoder, logit) - aponta
    exatamente em qual camada o valor nao-finito aparece pela primeira vez
  - no loop de treino real, torch.isfinite(logit).all() e checado logo
    apos o forward, antes de calcular a loss, pra confirmar se o logit ja
    chega nao-finito (problema no forward) ou so a loss que enlouquece

CAUSA RAIZ CONFIRMADA: amount/oldbalanceDest/newbalanceDest sem normalizacao
(valores na casa de milhoes) entrando direto no nn.Linear. Corrigido em
preprocessing/montar_event_stream.py - agora aplica z-score (calculado so
no split de treino) antes de salvar o event_stream.npz. Esta funcao _stats
por canal em EventStreamDataset._validar_dados() deve mostrar media~0,
std~1 para amount/delta_step/oldbalanceDest/newbalanceDest depois dessa
correcao; se ainda aparecer escala grande, o .npz foi gerado com a versao
antiga do script - regenere.

CORRIGIDO: o .npz salva um array "split" por conta ("treino"/"val"/"teste",
gerado por montar_event_stream.py a partir do mesmo split_por_conta() usado
no preprocess_pipeline.py). O treino agora usa esse campo em vez de
random_split() por exemplo - garante que a mesma conta nunca aparece em
treino e validacao ao mesmo tempo. O split "teste" e carregado mas
deliberadamente NAO usado no loop de treino/validacao - fica reservado para
avaliacao final (avaliacao/avaliar_modelo.py).

GPU com so 12% de uso durante o treino (gargalo no DataLoader, 1.9GB/8.1GB
de VRAM ocupados) - corrigido:
  - DataLoader com num_workers (--num-workers, default 4) + pin_memory=True
    (so quando device=="cuda") + persistent_workers - carrega batches em
    paralelo em vez de bloquear a GPU esperando o CPU montar cada batch
  - collate_fn trocado de lambda para functools.partial(montar_batch, ...):
    lambda nao e picklable, e no Windows num_workers>0 usa multiprocessing
    com metodo "spawn", que precisa picklar o collate_fn para os processos
    filhos - com lambda isso quebraria com PicklingError
  - --batch-size default subiu de 256 para 512 (VRAM sobrando); tente 1024
    se ainda sobrar
  - se der erro de multiprocessing no Windows com --num-workers 4, tente
    --num-workers 2 ou 0 (0 = sem subprocessos, como era antes)

AUPRC estagnado em 0.0016 apos 5 epocas (perto do piso = taxa de positivos,
0,12% -> 0.0016 e compativel com "modelo dando o mesmo score pra tudo").
Arquitetura atual: TransformerSequencial(d_model=64, n_heads=4, n_layers=3,
dim_feedforward=256, dropout=0.1). Duas mudancas:
  - avaliar() agora imprime, a cada epoca, a distribuicao de scores (media/
    mediana/min/max) separada por fraude vs legitima no split de validacao -
    se as duas distribuicoes vierem quase identicas, confirma que o modelo
    nao esta discriminando nada (nao e so a metrica AUPRC que esta ruim)
  - --lr default subiu de 1e-4 para 1e-3 - 1e-4 pode estar baixo demais pra
    sair do minimo trivial dado o tamanho do modelo/dataset. Rode:
    python transformer_sequencial.py --data ../../paysim_data/event_stream.npz --lr 1e-3
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent))
from positional_encoding import TimeAwarePositionalEncoding

N_TIPOS = 5  # CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER


class FocalLossComLogits(nn.Module):
    """
    Focal loss binaria (Lin et al., 2017) para logits - alternativa ao
    BCEWithLogitsLoss(pos_weight=...) para desbalanceamento extremo.

    A diferenca chave: pos_weight multiplica a perda da classe positiva por
    um fator FIXO e grande (aqui, ~836x pela taxa de fraude de 0.12%), o que
    estoura mesmo em fp32 quando o modelo erra feio no inicio do treino (log
    de uma probabilidade proxima de 0, vezes 836, facilmente vira inf). A
    focal loss em vez disso reduz a contribuicao dos exemplos "faceis" (ja
    bem classificados) de forma suave via (1 - p_t) ** gamma, sem depender de
    nenhum fator multiplicativo enorme - o termo alpha e so um balanceamento
    leve entre classes, tipicamente <= 1.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        perda = alpha_t * (1 - p_t).clamp(min=0) ** self.gamma * ce
        return perda.mean()


def _validar_dados(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Checa NaN/Inf antes do treino - causa comum de perda=nan logo na epoca 1.
    Roda UMA VEZ sobre o array inteiro (antes de fatiar por split), pra nao
    repetir os mesmos prints tres vezes quando os tres splits sao carregados.
    """
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan or n_inf:
        print(f"[aviso] event_stream.npz tem {n_nan:,} valores NaN e "
              f"{n_inf:,} valores Inf em X - substituindo NaN por 0 e "
              f"Inf por valor finito (nao resolve a causa raiz, so evita "
              f"que perda=nan trave o treino; considere revisar "
              f"montar_event_stream.py se a contagem for grande)")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    else:
        print("[validacao] X sem NaN/Inf.")

    y_float = y.astype("float32")
    n_nan_y = int(np.isnan(y_float).sum())
    if n_nan_y:
        raise ValueError(
            f"y tem {n_nan_y} valores NaN - dados de rotulo corrompidos, "
            f"verifique a geracao de event_stream.npz em montar_event_stream.py "
            f"antes de treinar (nao da para seguir com labels invalidos)."
        )

    # diagnostico: sem NaN/Inf nao significa sem valores extremos. Depois da
    # correcao do z-score em montar_event_stream.py, amount/delta_step/
    # oldbalanceDest/newbalanceDest devem aparecer com media~0, std~1 abaixo
    # - se ainda mostrarem escala grande (milhoes), o .npz foi gerado com a
    # versao antiga do script e precisa ser regenerado.
    print(f"dtype de X no disco (event_stream.npz): {X.dtype}")
    nomes_canais = ["tipo_idx", "amount", "delta_step", "oldbalanceDest", "newbalanceDest"]
    print("[diagnostico] estatisticas por canal em X (min/max/media/desvio):")
    for i, nome in enumerate(nomes_canais):
        canal = X[:, :, i]
        print(f"  {nome:16s} min={canal.min():>14.4f}  max={canal.max():>14.4f}  "
              f"media={canal.mean():>14.4f}  std={canal.std():>14.4f}")

    return X


def carregar_event_stream(npz_path: str):
    """
    Carrega event_stream.npz UMA VEZ e retorna os arrays completos (X, y,
    seq_lens) mais o array "split" (por conta - "treino"/"val"/"teste",
    gerado por montar_event_stream.py a partir do split_por_conta() de
    preprocess_pipeline.py). Usar esse campo para separar treino/val/teste
    evita o vazamento de conta que o random_split() por exemplo tinha.
    """
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]  # (N, janela, 5) -> [tipo_idx, amount, delta_step, old_bal, new_bal]
    y = data["y"]
    seq_lens = data["seq_lens"]

    if "split" not in data:
        raise ValueError(
            f"{npz_path} nao tem o campo 'split' - foi gerado com uma versao "
            f"antiga de montar_event_stream.py. Regenere o .npz antes de "
            f"treinar (necessario para separar treino/val/teste por conta, "
            f"sem vazamento)."
        )
    split = data["split"]

    X = _validar_dados(X, y)
    return X, y, seq_lens, split


class EventStreamDataset(Dataset):
    """
    Recebe os arrays JA fatiados para o split desejado. Nao carrega o .npz
    sozinho (isso e feito uma vez por carregar_event_stream(), e reutilizado
    para treino/val/teste) - evita reabrir o arquivo e revalidar NaN/Inf tres
    vezes.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_lens: np.ndarray):
        self.X = X
        self.y = y
        self.seq_lens = seq_lens

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        return {
            "tipo": torch.tensor(x[:, 0], dtype=torch.long),
            "numericas": torch.tensor(x[:, 1:], dtype=torch.float32),
            "seq_len": torch.tensor(self.seq_lens[idx], dtype=torch.long),
            "label": torch.tensor(self.y[idx], dtype=torch.float32),
        }


class TransformerSequencial(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 janela: int = 20, dropout: float = 0.1):
        super().__init__()
        self.embed_tipo = nn.Embedding(N_TIPOS, d_model // 2)
        self.proj_numericas = nn.Linear(3, d_model - d_model // 2)  # amount, old_bal, new_bal
        self.pos_encoding = TimeAwarePositionalEncoding(d_model, max_len=janela + 1)

        # token [CLS] aprendido, concatenado ao fim da sequencia
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        camada_encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(camada_encoder, num_layers=n_layers)
        self.classificador = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, tipo, numericas, delta_step, mascara_padding):
        # tipo: (B, L) | numericas: (B, L, 3) | delta_step: (B, L, 1)
        emb_tipo = self.embed_tipo(tipo)
        emb_num = self.proj_numericas(numericas)
        x = torch.cat([emb_tipo, emb_num], dim=-1)
        x = self.pos_encoding(x, delta_step)

        batch = x.size(0)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([x, cls], dim=1)  # cls no fim

        mascara_cls = torch.zeros(batch, 1, dtype=torch.bool, device=x.device)
        mascara_completa = torch.cat([mascara_padding, mascara_cls], dim=1)

        saida = self.encoder(x, src_key_padding_mask=mascara_completa)
        cls_saida = saida[:, -1, :]
        logit = self.classificador(cls_saida).squeeze(-1)
        return logit


@torch.no_grad()
def diagnosticar_primeiro_batch(modelo, tipo, numericas, delta_step, mascara_padding,
                                 device, usar_autocast: bool):
    """
    Roda o forward passo a passo (sem grad, nao interfere no treino real)
    so para o primeiro batch, imprimindo dtype e estatisticas de cada
    estagio. Objetivo: descobrir ONDE o valor nao-finito aparece pela
    primeira vez - se ja estiver em emb_num (projecao das features
    numericas), o problema e escala de entrada (amount/saldo sem
    normalizacao), nao a loss nem o pos_weight.
    """
    print("\n=== diagnostico do primeiro batch ===")
    print(f"dtype tipo: {tipo.dtype} | dtype numericas: {numericas.dtype} | "
          f"dtype delta_step: {delta_step.dtype}")
    print(f"device dos tensores de entrada: {tipo.device}")

    print(f"\nmodelo.embed_tipo.weight (pesos iniciais do embedding de tipo):")
    print(modelo.embed_tipo.weight)
    print(f"  finito? {torch.isfinite(modelo.embed_tipo.weight).all().item()} | "
          f"min={modelo.embed_tipo.weight.min().item():.4f} "
          f"max={modelo.embed_tipo.weight.max().item():.4f}")

    def _relatar(nome, tensor):
        finito = torch.isfinite(tensor).all().item()
        print(f"  [{nome}] dtype={tensor.dtype} finito={finito} "
              f"min={tensor.float().min().item():.4f} "
              f"max={tensor.float().max().item():.4f} "
              f"media={tensor.float().mean().item():.4f}")
        if not finito:
            print(f"    -> PRIMEIRO estagio com valor nao-finito encontrado: {nome}")
        return finito

    with torch.amp.autocast("cuda", enabled=usar_autocast):
        emb_tipo = modelo.embed_tipo(tipo)
        _relatar("emb_tipo", emb_tipo)

        emb_num = modelo.proj_numericas(numericas)
        _relatar("emb_num (projecao de amount/oldbalanceDest/newbalanceDest)", emb_num)

        x = torch.cat([emb_tipo, emb_num], dim=-1)
        _relatar("x (concat emb_tipo + emb_num, antes do positional encoding)", x)

        x = modelo.pos_encoding(x, delta_step)
        _relatar("x (depois do positional encoding)", x)

        batch_size = x.size(0)
        cls = modelo.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([x, cls], dim=1)

        mascara_cls = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        mascara_completa = torch.cat([mascara_padding, mascara_cls], dim=1)

        saida = modelo.encoder(x, src_key_padding_mask=mascara_completa)
        _relatar("saida (depois do TransformerEncoder)", saida)

        logit = modelo.classificador(saida[:, -1, :]).squeeze(-1)

    print(f"\nlogit (saida do forward, antes da loss): {logit}")
    print(f"logit finito? {torch.isfinite(logit).all().item()} | "
          f"dtype={logit.dtype} | min={logit.float().min().item():.4f} "
          f"max={logit.float().max().item():.4f}")
    print("=== fim do diagnostico ===\n")


def montar_batch(batch, janela):
    tipo = torch.stack([b["tipo"] for b in batch])
    numericas_full = torch.stack([b["numericas"] for b in batch])  # (B, L, 4): amount, delta, old, new
    delta_step = numericas_full[:, :, 1:2]
    numericas = torch.cat([numericas_full[:, :, 0:1], numericas_full[:, :, 2:4]], dim=-1)
    seq_lens = torch.stack([b["seq_len"] for b in batch])
    label = torch.stack([b["label"] for b in batch])

    posicoes = torch.arange(janela).unsqueeze(0)
    mascara_padding = posicoes >= (janela - seq_lens.unsqueeze(1))  # True = ignorar (padding no inicio)
    return tipo, numericas, delta_step, mascara_padding, label


def treinar(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"usando device: {device}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("[aviso] CUDA indisponivel - treinando em CPU. Se voce esperava "
              "usar a RTX 5060, confira `torch.cuda.is_available()` e o "
              "driver/instalacao do PyTorch com CUDA antes de continuar; "
              "141s/epoca em CPU seria esperado, na GPU nao deveria.")

    X, y, seq_lens, split = carregar_event_stream(args.data)

    idx_treino = np.where(split == "treino")[0]
    idx_val = np.where(split == "val")[0]
    idx_teste = np.where(split == "teste")[0]

    treino_ds = EventStreamDataset(X[idx_treino], y[idx_treino], seq_lens[idx_treino])
    val_ds = EventStreamDataset(X[idx_val], y[idx_val], seq_lens[idx_val])
    # teste carregado so para confirmar o tamanho - NAO entra em nenhum
    # DataLoader aqui. So avaliacao/avaliar_modelo.py deve toca-lo, e so
    # depois que o modelo estiver definido (avaliacao final, uma vez).
    n_teste = len(idx_teste)

    print(f"exemplos treino/val/teste: {len(treino_ds):,} / {len(val_ds):,} / {n_teste:,} "
          f"(split por conta - vindo de montar_event_stream.py, nao random_split)")
    if len(treino_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            "split 'treino' ou 'val' vazio no event_stream.npz - confira se "
            "montar_event_stream.py rodou corretamente (precisa do campo 'split')."
        )

    # functools.partial em vez de lambda: lambda nao e picklable, e no
    # Windows o DataLoader com num_workers>0 usa multiprocessing com
    # metodo "spawn", que PRECISA picklar o collate_fn para mandar aos
    # processos filhos - com lambda isso quebra com PicklingError.
    colar = functools.partial(montar_batch, janela=args.janela)

    usar_pin_memory = device == "cuda"
    print(f"DataLoader: num_workers={args.num_workers} pin_memory={usar_pin_memory} "
          f"batch_size={args.batch_size}")
    if args.num_workers > 0:
        print("  [aviso] no Windows, num_workers>0 as vezes da erro de "
              "multiprocessing (spawn) dependendo do ambiente - se travar ou "
              "estourar excecao, rode de novo com --num-workers 2 ou "
              "--num-workers 0.")

    treino_dl = DataLoader(
        treino_ds, batch_size=args.batch_size, shuffle=True, collate_fn=colar,
        num_workers=args.num_workers, pin_memory=usar_pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=colar,
        num_workers=args.num_workers, pin_memory=usar_pin_memory,
        persistent_workers=(args.num_workers > 0),
    )

    modelo = TransformerSequencial(janela=args.janela).to(device)

    if args.loss == "focal":
        criterio = FocalLossComLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"usando FocalLoss (alpha={args.focal_alpha}, gamma={args.focal_gamma}) "
              f"em vez de BCEWithLogitsLoss")
    else:
        with open(Path(__file__).parent.parent / "preprocessing" / "preprocessing_meta.json") as f:
            meta = json.load(f)
        # preprocessing_meta.json agora tem chaves separadas "tabular" (de
        # preprocess_pipeline.py) e "event_stream" (de montar_event_stream.py,
        # que ja inclui o z-score usado para normalizar X - ver EventStreamDataset)
        peso_bruto = meta.get("tabular", {}).get("class_weight_sugerido", {}).get("1") or 1.0
        peso_positivo = min(peso_bruto, args.pos_weight_max)
        print(f"pos_weight bruto (preprocessing_meta.json, ~1/taxa_fraude/2): {peso_bruto:.2f}")
        print(f"pos_weight usado no treino (capado em {args.pos_weight_max}): {peso_positivo:.2f}")
        if peso_bruto > args.pos_weight_max:
            print(f"  [aviso] pos_weight bruto ({peso_bruto:.2f}) excede o teto - "
                  f"foi capado. Se a perda ainda vier nao-finita mesmo assim, "
                  f"rode com --loss focal (ver --help).")
        criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(peso_positivo, device=device))

    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr)
    # sintaxe nova (torch.amp.*, com device_type explicito) - torch.cuda.amp.*
    # esta deprecated a partir do PyTorch 2.x
    scaler_amp = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    melhor_auprc = -1.0
    checkpoint_path = Path(__file__).parent / "checkpoint_melhor_modelo.pt"
    diagnostico_ja_feito = False

    for epoca in range(args.epocas):
        modelo.train()
        inicio = time.time()
        perda_total = 0.0
        n_batches_validos = 0
        n_batches_pulados = 0
        for tipo, numericas, delta_step, mascara, label in treino_dl:
            tipo, numericas, delta_step = tipo.to(device), numericas.to(device), delta_step.to(device)
            mascara, label = mascara.to(device), label.to(device)

            if not diagnostico_ja_feito:
                diagnosticar_primeiro_batch(
                    modelo, tipo, numericas, delta_step, mascara,
                    device, usar_autocast=(device == "cuda"),
                )
                diagnostico_ja_feito = True

            otimizador.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logit = modelo(tipo, numericas, delta_step, mascara)

            # diagnostico pedido: confirma se o problema ja esta no forward
            # (antes da loss) ou so aparece depois de calcular a loss
            logit_finito = torch.isfinite(logit).all()
            if not logit_finito:
                print(f"  [diagnostico] logit NAO finito antes da loss "
                      f"(epoca {epoca+1}) - o problema esta no forward "
                      f"(embeddings/projecao/encoder), nao na funcao de perda")

            # BCEWithLogitsLoss com pos_weight alto (desbalanceamento ~0.13%)
            # calculada FORA do autocast, em fp32: em fp16 o produto
            # perda * pos_weight * escala_do_GradScaler estoura com
            # facilidade e vira inf/nan logo na primeira epoca - essa e a
            # causa mais provavel do perda=nan reportado.
            perda = criterio(logit.float(), label)

            if not torch.isfinite(perda):
                n_batches_pulados += 1
                print(f"  [aviso] perda nao finita neste batch (epoca {epoca+1}) - "
                      f"pulando, sem atualizar pesos")
                continue

            scaler_amp.scale(perda).backward()
            # grad clipping - precisa "unscale" antes de calcular a norma,
            # senao os gradientes estao multiplicados pela escala do AMP
            scaler_amp.unscale_(otimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            scaler_amp.step(otimizador)
            scaler_amp.update()
            perda_total += perda.item()
            n_batches_validos += 1

        perda_media = perda_total / max(n_batches_validos, 1)
        auprc_val = avaliar(modelo, val_dl, device)
        tempo_epoca = time.time() - inicio
        print(f"epoca {epoca+1}/{args.epocas} | perda {perda_media:.4f} "
              f"| auprc_val {auprc_val:.4f} | {tempo_epoca:.1f}s "
              f"| batches pulados por perda nao finita: {n_batches_pulados}")
        if n_batches_pulados > 0:
            print(f"  [aviso] {n_batches_pulados} batches pulados nesta epoca - "
                  f"se isso persistir por varias epocas, o lr ainda pode estar "
                  f"alto ou ha valores extremos nos dados (ver aviso de "
                  f"validacao no inicio do treino)")

        if auprc_val > melhor_auprc:
            melhor_auprc = auprc_val
            torch.save({
                "model_state": modelo.state_dict(),
                "auprc_val": auprc_val,
                "epoca": epoca,
                "args": vars(args),
            }, checkpoint_path)
            print(f"  -> novo melhor modelo salvo em {checkpoint_path}")

    print(f"treino concluido. melhor AUPRC val: {melhor_auprc:.4f}")


@torch.no_grad()
def avaliar(modelo, dl, device, relatar_distribuicao: bool = True):
    from sklearn.metrics import average_precision_score

    modelo.eval()
    todos_logits, todos_labels = [], []
    for tipo, numericas, delta_step, mascara, label in dl:
        tipo, numericas, delta_step = tipo.to(device), numericas.to(device), delta_step.to(device)
        mascara = mascara.to(device)
        logit = modelo(tipo, numericas, delta_step, mascara)
        todos_logits.append(torch.sigmoid(logit).cpu())
        todos_labels.append(label)

    probs = torch.cat(todos_logits).numpy()
    labels = torch.cat(todos_labels).numpy()

    if relatar_distribuicao:
        # AUPRC estagnado perto do piso (~taxa de positivos) geralmente
        # significa que o modelo da o MESMO score pra tudo (nao aprendeu
        # nada) - separar scores de fraude vs legitimo mostra isso direto:
        # se as duas distribuicoes forem quase identicas, o modelo nao esta
        # discriminando; se fraude tiver scores mais altos mas o threshold/
        # AUPRC ainda vier ruim, o problema e outro (calibração, dados, etc).
        probs_fraude = probs[labels == 1]
        probs_legit = probs[labels == 0]
        print(f"  [scores val] fraude    (n={len(probs_fraude):>6}): "
              f"media={probs_fraude.mean():.4f}  mediana={np.median(probs_fraude):.4f}  "
              f"min={probs_fraude.min():.4f}  max={probs_fraude.max():.4f}"
              if len(probs_fraude) > 0 else "  [scores val] fraude: nenhum exemplo neste batch/split")
        print(f"  [scores val] legitima  (n={len(probs_legit):>6}): "
              f"media={probs_legit.mean():.4f}  mediana={np.median(probs_legit):.4f}  "
              f"min={probs_legit.min():.4f}  max={probs_legit.max():.4f}"
              if len(probs_legit) > 0 else "  [scores val] legitima: nenhum exemplo neste batch/split")
        if len(probs_fraude) > 0 and len(probs_legit) > 0:
            diferenca_medias = probs_fraude.mean() - probs_legit.mean()
            print(f"  [scores val] diferenca de media (fraude - legitima): {diferenca_medias:+.4f} "
                  f"{'(quase zero -> modelo NAO esta discriminando)' if abs(diferenca_medias) < 0.01 else ''}")

    if labels.sum() == 0:
        return 0.0
    return average_precision_score(labels, probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                         help="caminho para event_stream.npz gerado por montar_event_stream.py")
    parser.add_argument("--janela", type=int, default=13,
                         help="p90 de transacoes por nameDest (analisar_namedest.py)")
    parser.add_argument("--batch-size", type=int, default=512,
                         help="512 por padrao (era 256) - GPU estava com so "
                              "12%% de uso e 1.9GB/8.1GB de VRAM ocupados; "
                              "tente 1024 se ainda sobrar VRAM")
    parser.add_argument("--num-workers", type=int, default=4,
                         help="processos do DataLoader para carregar batches "
                              "em paralelo - gargalo identificado (GPU ociosa "
                              "esperando dado). No Windows, se der erro de "
                              "multiprocessing/pickling, tente 2 ou 0")
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="1e-3 (subiu de 1e-4) - AUPRC estagnado em 0.0016 "
                              "por 5 epocas sugere que 1e-4 pode estar baixo "
                              "demais pra sair do minimo trivial (score igual "
                              "pra tudo). Se voltar a dar perda nao-finita, "
                              "recue para 5e-4 ou 1e-4.")
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce",
                         help="bce = BCEWithLogitsLoss com pos_weight capado "
                              "(--pos-weight-max); focal = FocalLoss, use se "
                              "'bce' ainda gerar perda nao-finita")
    parser.add_argument("--pos-weight-max", type=float, default=50.0,
                         help="teto para o pos_weight do BCEWithLogitsLoss - "
                              "o valor bruto (~836x pela taxa de fraude de "
                              "0.12%%) estoura mesmo em fp32")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    args = parser.parse_args()
    treinar(args)
