# Script do Desafio — Detecção de Fraude com Event Stream (PaySim)
> Segundo projeto prático — evolução do credit card fraud para LDM com sequência temporal real.
> Dataset: PaySim Mobile Money (Kaggle — ntnu-testimon/paysim1)
> Diferencial: tem clienteID → permite montar event stream por cliente → transformer sequencial real

---

## Contexto de Negócio (simular com o cliente)

**Cliente simulado:** operadora de mobile money africana expandindo para o Brasil

**Problema:**
- Fraudes em transferências e saques crescendo 25% ao mês
- Modelo atual: regras baseadas em valor e tipo de transação
- Limitação: não considera o histórico sequencial do cliente
- Dataset: 6.3 milhões de transações com clienteID — event stream disponível

**Diferença fundamental para o projeto anterior:**
```
credit card:  sem clienteID → atenção entre features de 1 transação
PaySim:       com clienteID → sequência de eventos por cliente
                              → transformer sequencial (LDM de verdade)
                              → pode detectar padrões que precedem fraude
```

---

## Pré-requisitos — Instalar antes de começar

```bash
# 1. PyTorch com suporte CUDA (RTX 5090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. verificar se GPU está ativa
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# deve retornar: True | NVIDIA GeForce RTX 5090

# 3. bibliotecas de ML
pip install pandas scikit-learn matplotlib seaborn

# 4. monitoramento de GPU
pip install nvitop        # monitor em tempo real no terminal
pip install wandb         # dashboard online com relatórios
pip install pynvml        # acesso programático às métricas da GPU
pip install gpustat       # relatório rápido de status da GPU

# 5. baixar dataset
# kaggle.com/datasets/ntnu-testimon/paysim1
# arquivo: PS_20174392719_1491204439457_log.csv (~470MB)
```

---

## Ferramentas de Monitoramento de GPU

### Durante o treino — abrir em terminal separado

```bash
# opção 1: nvidia-smi (já instalado com o driver)
nvidia-smi dv 1          # atualiza a cada 1 segundo

# opção 2: nvitop (interface visual mais rica)
nvitop                   # mostra GPU, VRAM, temperatura, processos

# opção 3: gpustat (compacto, bom para logs)
gpustat --watch 1        # atualiza a cada 1 segundo
```

### Dentro do código Python — loga durante o treino

```python
import torch
import pynvml

# inicializa monitoramento
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def gpu_stats():
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # watts
    return {
        'vram_usada_gb':   mem.used / 1e9,
        'vram_total_gb':   mem.total / 1e9,
        'temperatura_c':   temp,
        'uso_gpu_pct':     util.gpu,
        'consumo_watts':   power,
    }

# usar no loop de treino
for epoca in range(epocas):
    stats = gpu_stats()
    print(f"época {epoca} | "
          f"VRAM {stats['vram_usada_gb']:.1f}/{stats['vram_total_gb']:.0f}GB | "
          f"temp {stats['temperatura_c']}°C | "
          f"GPU {stats['uso_gpu_pct']}% | "
          f"{stats['consumo_watts']:.0f}W")
```

### wandb — dashboard online com histórico completo

```python
import wandb

wandb.init(project="paysim-ldm", name="transformer-sequencial-v1")

# loga a cada época
wandb.log({
    "loss":           loss.item(),
    "auprc":          auprc,
    "recall_fraude":  recall,
    "vram_gb":        torch.cuda.memory_allocated()/1e9,
    "temperatura_c":  stats['temperatura_c'],
    "uso_gpu_pct":    stats['uso_gpu_pct'],
    "consumo_watts":  stats['consumo_watts'],
    "epoch":          epoca,
    "tokens_por_seg": tokens_processados / tempo_epoca,
})

wandb.finish()
```

### Parâmetros importantes de observar

```
temperatura:      ideal < 83°C em carga
                  acima de 90°C → throttling (GPU reduz clock)
                  RTX 5090: limite ~90°C

uso da GPU (%):   ideal > 90% durante treino
                  se baixo → gargalo no CPU ou dataloader
                  solução: aumentar batch size ou num_workers

VRAM usada:       acompanhar para não estourar 32GB
                  PaySim com sequências longas pode ser pesado
                  se estourar → reduzir batch size ou comprimento da sequência

consumo (Watts):  RTX 5090 pode chegar a 575W em carga total
                  normal para treino intensivo

tokens/segundo:   métrica de throughput do transformer
                  indica eficiência do pipeline de dados
```

---

## Estrutura do Projeto

```
tt-docs/projetos/paysim/
├── eda/
│   ├── EDA_paysim.md
│   ├── viz_distribuicao_tipos.png
│   ├── viz_sequencia_cliente.png      ← novo: visualização de event stream
│   ├── viz_padroes_temporais.png      ← novo: padrões antes da fraude
│   └── viz_separabilidade.png
│
├── preprocessing/
│   ├── preprocess_pipeline.py
│   ├── montar_event_stream.py         ← novo: agrupa transações por cliente
│   ├── preprocessing_meta.json
│   └── limpeza_e_eventstream.md
│
├── modelo/
│   ├── transformer_sequencial.py      ← diferente: atenção entre transações
│   ├── positional_encoding.py         ← novo: posição dos eventos no tempo
│   └── checkpoint_melhor_modelo.pt
│
├── avaliacao/
│   ├── curva_pr.png
│   ├── matriz_confusao.png
│   ├── resultado_final.md
│   └── gpu_report.md                  ← novo: relatório de uso da GPU
│
├── gpu_monitoring/
│   ├── wandb_config.py                ← configuração do dashboard
│   ├── gpu_logger.py                  ← logger de métricas da GPU
│   └── relatorio_gpu.png              ← gráficos de temperatura, VRAM, uso
│
└── README.md
```

---

## Passo a Passo do Projeto

### FASE 1 — Contexto de Negócio
```
[ ]  simular reunião com cliente (você apresenta o problema)
[ ]  identificar o problema central e impacto financeiro
[ ]  definir critérios de sucesso (recall, FPR, threshold)
[ ]  identificar diferença fundamental para o projeto anterior
     (event stream vs tabular)
```

### FASE 2 — EDA
```
[ ]  distribuição de tipos de transação (CASH_IN, CASH_OUT, TRANSFER...)
[ ]  desbalanceamento (fraude é ~0.13%)
[ ]  visualizar sequência de eventos de um cliente específico
     → quantas transações tem um cliente em média?
     → qual o padrão antes de uma fraude?
[ ]  padrão temporal: fraudes concentradas em algum horário?
[ ]  análise de valor por tipo de transação
[ ]  K-Means nos clientes fraudadores — tipos de fraude
[ ]  decidir comprimento da janela de contexto (últimas N transações)
```

### FASE 3 — Pré-processamento
```
[ ]  remover duplicatas
[ ]  montar event stream por cliente
     → ordenar transações por tempo para cada clienteID
     → criar sequências de comprimento fixo (ex: últimas 10 transações)
     → padding para clientes com menos transações
[ ]  encoding dos tipos de transação (one-hot ou embedding)
[ ]  z-score nos valores numéricos (amount, saldo antes/depois)
[ ]  positional encoding para posição na sequência
[ ]  split estratificado por cliente (não por transação)
     → evita que o mesmo cliente esteja em treino e teste
[ ]  class_weight para desbalanceamento
```

### FASE 4 — Configurar Monitoramento de GPU
```
[ ]  abrir nvitop em terminal separado
[ ]  configurar wandb (criar conta gratuita em wandb.ai)
[ ]  inicializar gpu_logger.py no código
[ ]  testar: rodar 1 época e verificar se métricas aparecem no dashboard
```

### FASE 5 — Arquitetura do Transformer Sequencial
```
[ ]  decisão: quantos eventos na sequência de contexto?
[ ]  embedding de tipo de transação (CASH_OUT, TRANSFER...)
[ ]  positional encoding (posição no tempo)
[ ]  N blocos de encoder com self-attention entre eventos
     → diferente do projeto anterior (era atenção entre features)
     → agora é atenção entre transações do mesmo cliente
[ ]  token [CLS] para classificação
[ ]  sigmoid na saída → score de fraude
[ ]  ativar CUDA: model.to('cuda')
```

### FASE 6 — Treino com GPU
```
[ ]  monitorar temperatura (manter < 83°C)
[ ]  monitorar VRAM (não estourar 32GB)
[ ]  early stopping por AUPRC de validação
[ ]  salvar checkpoints
[ ]  comparar tempo por época vs projeto anterior no CPU
[ ]  gerar relatório de GPU no wandb
```

### FASE 7 — Avaliação
```
[ ]  AUPRC no conjunto de validação
[ ]  definir threshold com cliente
[ ]  recall no tipo de fraude mais custoso
[ ]  avaliar no teste (uma única vez)
[ ]  comparativo: modelo anterior (tabular) vs este (sequencial)
[ ]  demonstrar: o modelo detecta card testing antes da fraude?
```

### FASE 8 — Relatório de GPU
```
[ ]  exportar dashboard do wandb
[ ]  temperatura máxima atingida
[ ]  VRAM utilizada vs disponível
[ ]  consumo médio em Watts
[ ]  tokens processados por segundo
[ ]  comparativo: quanto tempo levaria no CPU?
[ ]  salvar em avaliacao/gpu_report.md
```

### FASE 9 — Síntese e Próximos Passos
```
[ ]  apresentar resultado ao cliente
[ ]  comparar com projeto anterior (credit card)
[ ]  argumentar: por que event stream é melhor?
[ ]  propor evolução: fine-tuning com LoRA/QLoRA
[ ]  shadow mode e gradual rollout
[ ]  atualizar notas do desafio e casos_de_uso.md
```

---

## Diferenças Técnicas vs Projeto Anterior

| Aspecto | Credit Card (anterior) | PaySim (este) |
|---|---|---|
| Sequência | sem clienteID | com clienteID |
| Atenção | entre features | entre transações |
| Tokens | 1 por feature | 1 por evento/transação |
| Positional encoding | não necessário | sim (ordem temporal importa) |
| Padding | não | sim (sequências de tamanho variável) |
| Split | por transação | por cliente |
| Hardware | CPU (Cowork) | RTX 5090 (local) |
| Monitoramento | não | nvitop + wandb + pynvml |

---

## Meta do Projeto

```
técnica:   AUPRC > 0.80 (melhor que o projeto anterior: 0.705)
negócio:   detectar card testing ANTES da fraude principal
           demonstrar que event stream > features isoladas
gpu:       treino completo em < 10 minutos na RTX 5090
relatório: dashboard wandb com histórico completo de GPU
```