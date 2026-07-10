# Troubleshooting — Projeto PaySim Transformer
> Registro dos problemas encontrados durante a implementação e as soluções aplicadas.
> Objetivo: aprender com os erros e ter referência para projetos futuros.

---

## Problema 1 — PyTorch não instalava (Python 3.14)

**Sintoma:**
```
ERROR: Could not find a version that satisfies the requirement torch
ERROR: No matching distribution found for torch
```

**Causa:**
Python 3.14 é muito novo — PyTorch suporta até Python 3.12 oficialmente.

**Solução:**
```bash
winget install Python.Python.3.12
py -3.12 -m pip install torch ...
```

**Lição:**
Sempre verificar a compatibilidade de versão entre Python e PyTorch antes de instalar.
Referência: pytorch.org/get-started/locally

---

## Problema 2 — PyTorch instalado mas GPU não reconhecida (RTX 5060)

**Sintoma:**
```
UserWarning: NVIDIA GeForce RTX 5060 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 ... sm_90.
```

**Causa:**
RTX 5060 usa arquitetura Blackwell (sm_120). PyTorch 2.5.1 com CUDA 12.1 não suporta Blackwell — só até Hopper (sm_90).

**Solução:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# PyTorch 2.11.0 + CUDA 12.8 = suporte a Blackwell
```

**Lição:**
GPUs RTX 50xx (Blackwell) precisam de CUDA 12.8 + PyTorch 2.6+.
RTX 40xx (Ada) funciona com CUDA 12.1 + PyTorch 2.5.

---

## Problema 3 — perda=nan na época 1

**Sintoma:**
```
epoca 1/20 | perda nan | auprc_val 0.0012 | 141.3s
```

**Causas identificadas (em ordem de investigação):**

**3a. autocast deprecated:**
```python
# errado (deprecated no PyTorch 2.11)
with torch.cuda.amp.autocast(enabled=...):

# correto
with torch.amp.autocast('cuda', enabled=...):
```

**3b. loss calculada dentro do autocast em fp16:**
```python
# errado — loss em fp16 com pos_weight alto causa overflow
with torch.amp.autocast('cuda'):
    logit = model(X)
    loss = criterio(logit, label)  # overflow aqui

# correto — loss em fp32
with torch.amp.autocast('cuda'):
    logit = model(X)
loss = criterio(logit.float(), label)  # fora do autocast = fp32
```

**3c. pos_weight muito alto (~836x):**
```
desbalanceamento: 0.12% de fraudes → ratio 1:836
pos_weight bruto: 836  → overflow mesmo em fp32
solução: capar pos_weight em 50.0

pos_weight bruto (168.66) → capado em 50.0
```

**3d. Focal Loss como alternativa ao pos_weight alto:**
```python
# BCEWithLogitsLoss com pos_weight alto = instável
# FocalLoss = mais estável para desbalanceamento extremo
criterio = FocalLossComLogits(alpha=0.25, gamma=2.0)
```

**Lição geral sobre perda=nan:**
```
checklist quando perda=nan:
1. dados têm NaN/Inf?  → torch.isfinite(X).all()
2. loss dentro do autocast? → mover para fora
3. pos_weight muito alto? → capar em 50 ou usar FocalLoss
4. learning rate alto? → reduzir para 1e-4 ou menos
5. overflow nos embeddings? → verificar dtype das entradas
6. valores extremos nos dados? → verificar min/max das features
```

---

## Problema 4 — 141s por época (GPU não sendo usada)

**Sintoma:**
```
epoca 1/20 | perda nan | auprc_val 0.0012 | 141.3s
```
141 segundos por época é tempo de CPU — GPU deveria ser 5-15s.

**Causa:**
O modelo ou os dados não estavam sendo movidos para o device CUDA.

**Diagnóstico:**
```python
# adicionar no início do treino
print(f"usando device: {device}")
# deve retornar: usando device: cuda
```

**Solução:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X = X.to(device)
y = y.to(device)
```

**Lição:**
Sempre confirmar que GPU está sendo usada com print do device.
Se tempo por época for muito alto → provavelmente CPU rodando.

---

## Problema 5 — perda=nan persiste mesmo com Focal Loss (em investigação)

**Sintoma:**
```
usando FocalLoss (alpha=0.25, gamma=2.0)
[aviso] perda nao finita neste batch (epoca 1) - pulando
```

**Status:** em investigação

**Hipóteses:**
```
1. overflow nos embeddings ou inicialização dos pesos
2. valores extremos em amount/oldbalanceDest/newbalanceDest
   (transações podem ter valores muito altos no PaySim)
3. dtype incorreto das entradas (X chegando como float16?)
4. problema no forward pass antes da loss
```

**Diagnóstico em andamento:**
```python
# verificar saída do forward antes da loss
logit = model(X)
print(torch.isfinite(logit).all())  # deve ser True

# verificar dtype das entradas
print(X.dtype)  # deve ser float32

# verificar valores extremos
print(f"amount max: {X[:,:,1].max()}")
print(f"balance max: {X[:,:,3].max()}")
```

**Lição parcial:**
Quando dados estão limpos (sem NaN/Inf) mas perda ainda é nan,
o problema está no forward pass — embeddings, ativações ou valores extremos
nos dados que causam overflow nas operações matriciais.

---

## Conceitos Aprendidos

**fp16 vs fp32:**
```
fp16 (half precision):  menor memória, mais rápido na GPU
                        range limitado: ~65.000 máximo
                        operações com valores grandes → overflow → nan

fp32 (full precision):  mais memória, mais lento
                        range maior: ~3.4 × 10^38
                        mais estável para loss e gradientes
```

**autocast (precisão mista):**
```
combina fp16 (forward pass) com fp32 (loss e gradientes)
melhor dos dois mundos: velocidade + estabilidade
regra: forward dentro do autocast, loss FORA
```

**GradScaler:**
```
escala a loss para evitar underflow dos gradientes em fp16
deve fazer unscale_ antes do gradient clipping:
  scaler.unscale_(optimizer)
  clip_grad_norm_(model.parameters(), max_norm=1.0)
  scaler.step(optimizer)
  scaler.update()
```

**pos_weight vs FocalLoss:**
```
pos_weight:   multiplica a loss da classe positiva
              intuitivo mas instável com valores altos (>50)
              BCEWithLogitsLoss(pos_weight=tensor([50.0]))

FocalLoss:    reduz a contribuição de exemplos fáceis
              (1-p_t)^gamma  onde gamma=2.0 é comum
              mais estável para desbalanceamento extremo
              não depende de um fator multiplicativo grande
```

**Gradient clipping:**
```
limita o tamanho máximo dos gradientes
evita exploding gradient em transformers profundos
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Atualização — Problema 5 RESOLVIDO (causa identificada)

**Causa confirmada:**
```
logit NAO finito antes da loss
→  problema está no forward pass, não na função de perda
→  hipótese do Cowork confirmada:
   amount, oldbalanceDest, newbalanceDest chegam SEM normalização
   valores na casa de dezenas de milhões
   nn.Linear aplicado a valores tão grandes → overflow → nan
```

**Por que FocalLoss também falhou:**
```
trocar a loss não resolveu porque o problema era ANTERIOR à loss
o logit já chegava não-finito antes de qualquer cálculo de perda
→  sempre diagnosticar o forward antes de trocar a loss
```

**Solução:**
```python
# em montar_event_stream.py ou preprocess_pipeline.py
# aplicar z-score nas features numéricas ANTES de salvar no .npz

# calcular estatísticas APENAS no treino
media_amount  = X_treino[:,:,1].mean()
desvio_amount = X_treino[:,:,1].std()

# aplicar nos três conjuntos
X_treino[:,:,1] = (X_treino[:,:,1] - media_amount) / desvio_amount
X_val[:,:,1]    = (X_val[:,:,1]    - media_amount) / desvio_amount
X_teste[:,:,1]  = (X_teste[:,:,1]  - media_amount) / desvio_amount

# salvar estatísticas no meta.json para inferência em produção
```

**Features que precisam de z-score:**
```
amount          →  valores até dezenas de milhões
oldbalanceDest  →  mesma escala
newbalanceDest  →  mesma escala
delta_step      →  pode ter valores grandes dependendo do gap temporal
```

**Lição fundamental:**
```
NUNCA alimentar um nn.Linear com features sem normalização
especialmente quando os dados têm valores em escalas muito diferentes

checklist de normalização para transformers:
  features categóricas  →  embedding (não normaliza)
  features numéricas    →  z-score ANTES de entrar no modelo
  posição temporal      →  normalizar ou usar positional encoding relativo

o z-score deve sempre ser calculado no treino e aplicado nos demais conjuntos
as estatísticas devem ser salvas para uso em produção (inferência)
```

**Diagrama do problema:**
```
amount = 9.000.000 (sem normalização)
        ↓
nn.Linear (projeção numérica)
        ↓
valor intermediário → overflow → nan
        ↓
encoder layers → nan se propaga
        ↓
logit = nan
        ↓
loss = nan  (independente de qual função de perda)
```

---

## Problema 6 — Data Leakage por conta no split do transformer (pendente)

**Sintoma identificado pelo Cowork:**
```
montar_event_stream.py: split correto por conta (sem vazamento)
transformer_sequencial.py: ainda usa random_split() por exemplo
→  mesma conta pode aparecer em treino E validação
→  data leakage via sequência de conta
```

**Por que isso é um problema:**
```
split por EXEMPLO (errado):
  sequência 1 da conta X → treino
  sequência 2 da conta X → validação
  modelo "vê" a conta X no treino e é avaliado na mesma conta
  → AUPRC inflado artificialmente

split por CONTA (correto):
  conta X → todas as sequências vão para treino OU validação
  conta Y → todas as sequências vão para validação OU teste
  → avaliação honesta em contas nunca vistas
```

**Status:** documentado, aguardando decisão de corrigir ou não

**Impacto no projeto:**
```
se não corrigir:  AUPRC de validação vai parecer melhor que é
                  modelo pode não generalizar para contas novas
                  em produção vai atender contas novas — precisa generalizar

se corrigir:      precisa rodar passo 1 e passo 3 novamente
                  resultado mais honesto e seguro para produção
```

**Lição:**
```
em projetos com sequências por entidade (cliente, conta, usuário):
  SEMPRE fazer split por entidade, não por exemplo
  regra: a mesma entidade nunca pode aparecer em treino E validação
  igual ao que fizemos no credit card: split estratificado por conta
```

---

## Problema 5 — RESOLVIDO ✅

**Solução aplicada:**
```
z-score calculado só no split de treino e aplicado nos três conjuntos
estatísticas salvas em preprocessing_meta.json (chave 'event_stream')

amount:          média 318.312  desvio 895.488
oldbalanceDest:  média 1.7M     desvio 4.2M
newbalanceDest:  média 2.0M     desvio 4.7M
delta_step:      média 30       desvio 51
```

**Confirmação:**
```
época 1: perda 0.0012 (finita!) | batches pulados: 0
problema de nan resolvido completamente
```

---

## Problema 7 — GPU com baixa utilização (gargalo no DataLoader)

**Sintoma:**
```
nvitop mostra:
  GPU Util:    12%      ← deveria estar > 80%
  VRAM:        1.9GB / 8.1GB  ← usando pouco
  Temperatura: 42°C    ← GPU fria = não está trabalhando
  Consumo:     39W / 145W  ← muito abaixo do máximo
  CPU:         15.8%   ← CPU fazendo mais trabalho que GPU
  tempo/época: 141s    ← igual ao CPU — GPU não está acelerando
```

**Causa:**
```
DataLoader com num_workers=0 (default)
GPU fica esperando os dados chegarem do CPU
o carregamento e pré-processamento dos batches é o gargalo
→  GPU subutilizada
```

**Solução:**
```python
# aumentar num_workers para paralelizar o carregamento
DataLoader(dataset, batch_size=..., num_workers=4, pin_memory=True)

# pin_memory=True: dados carregados direto na memória pinada
# mais rápido para transferir CPU → GPU

# aumentar batch_size aproveitando a VRAM disponível
# estava usando 1.9GB de 8.1GB → pode aumentar batch_size
# tentar: batch_size=512 ou 1024
```

**Atenção para Windows:**
```
num_workers > 0 no Windows pode causar problemas com multiprocessing
se der erro: usar num_workers=2 ou adicionar
if __name__ == '__main__': no script principal
```

**Impacto esperado:**
```
antes:  141s/época, GPU 12%
depois: 20-40s/época, GPU > 80%
```

**Lição:**
```
GPU subutilizada durante treino quase sempre é gargalo no DataLoader
checklist:
  num_workers > 0      → paraleliza carregamento
  pin_memory=True      → transferência CPU→GPU mais rápida
  batch_size adequado  → aproveita a VRAM disponível
  prefetch_factor=2    → pré-carrega próximo batch enquanto GPU processa atual
```

**Status:** aguardando correção do Cowork

---

## Problema 7 — RESOLVIDO ✅ (DataLoader gargalo)

**Solução aplicada:**
```python
DataLoader(dataset, batch_size=512, num_workers=4, pin_memory=True)
```
**Resultado:** tempo por época caiu de 141s para 60-75s

---

## Problema 8 — Modelo sem sinal discriminativo (RESOLVIDO por diagnóstico)

**Sintoma:**
```
scores val fraude:    média 0.0487
scores val legítimo:  média 0.0486
diferença:            0.0001  ← modelo não discrimina nada
AUPRC:                0.0016  ← aleatório
```

**Causa raiz identificada:**
```
sinal de fraude no PaySim está no REMETENTE:
  erroBalanceOrig  →  remetente drena 100% do saldo (63.9% da importância)
  newbalanceOrig   →  saldo vai a zero (28.1% da importância)

quando pivotamos para nameDest como âncora do event stream:
  paramos de carregar oldbalanceOrg/newbalanceOrig/nameOrig
  modelo ficou cego para 92% do sinal de fraude
  as features disponíveis (saldo do destinatário) não separam as classes
```

**Confirmação via baseline XGBoost:**
```
AUPRC baseline (com features do remetente):  0.9972  ← quase perfeito
AUPRC transformer (sem features remetente):  0.0016  ← aleatório

top features:
  erroBalanceOrig:  63.9%  ← domina completamente
  newbalanceOrig:   28.1%  ← segunda mais importante
```

**Conclusão sobre o PaySim:**
```
o padrão de fraude do PaySim é uma regra de 1 transação
erroBalanceOrig ≈ 1  →  fraude com 99.72% de certeza
não precisa de sequência temporal
não precisa de transformer
→  PaySim não é o dataset ideal para demonstrar event stream sequencial
```

**Decisão:**
```
aceitar o baseline XGBoost (AUPRC 0.9972) como resultado final
documentar a limitação do PaySim
migrar para dataset sintético com padrão genuinamente sequencial
```

**Lição fundamental:**
```
SEMPRE rodar um baseline simples ANTES de implementar o transformer

checklist antes de usar transformer sequencial:
  1. o padrão realmente requer contexto histórico?
  2. uma transação isolada com as features certas já detecta?
  3. o sinal está nas features que estou alimentando ao modelo?
  4. baseline tabular confirma que há sinal discriminativo?

se baseline simples tem AUPRC > 0.95 → o problema não precisa de transformer
se baseline simples tem AUPRC baixo  → aí vale investigar arquitetura sequencial
```