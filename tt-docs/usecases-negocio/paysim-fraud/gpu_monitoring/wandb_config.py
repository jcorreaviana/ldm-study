"""
Configuracao do Weights & Biases para o projeto paysim-ldm.

Requer conta gratuita em wandb.ai e `wandb login` no terminal (ou variavel
de ambiente WANDB_API_KEY) antes de rodar - nao incluido/hardcoded aqui.

Uso:

    from wandb_config import iniciar_wandb, logar_epoca

    run = iniciar_wandb(nome_run="transformer-sequencial-v1", config=vars(args))
    for epoca in range(epocas):
        ...
        logar_epoca(epoca, loss=perda.item(), auprc=auprc_val, gpu_stats=stats)
    run.finish()
"""

import wandb

PROJETO = "paysim-ldm"


def iniciar_wandb(nome_run: str, config: dict, projeto: str = PROJETO):
    return wandb.init(project=projeto, name=nome_run, config=config)


def logar_epoca(epoca: int, loss: float, auprc: float | None = None,
                 recall: float | None = None, gpu_stats: dict | None = None,
                 tokens_por_seg: float | None = None):
    payload = {"epoch": epoca, "loss": loss}
    if auprc is not None:
        payload["auprc"] = auprc
    if recall is not None:
        payload["recall_fraude"] = recall
    if tokens_por_seg is not None:
        payload["tokens_por_seg"] = tokens_por_seg
    if gpu_stats:
        payload.update({
            "vram_gb": gpu_stats.get("vram_usada_gb"),
            "temperatura_c": gpu_stats.get("temperatura_c"),
            "uso_gpu_pct": gpu_stats.get("uso_gpu_pct"),
            "consumo_watts": gpu_stats.get("consumo_watts"),
        })
    wandb.log(payload)
