# EDA — PaySim (mobile money) — event stream por conta de destino

Contexto: operadora de mobile money, fraude crescendo (script-desafio.md fala
em 25%/mês; a conversa inicial com José mencionou 180% em 3 meses — confirmar
o número real com o cliente). Modelo atual bloqueia por valor e tipo de
transação; não considera histórico sequencial.

**Nota metodológica sobre o sandbox**: os números desta EDA vêm de duas
fontes. Os itens 1 a 6 (shape, tipos, desbalanceamento, fraude por tipo,
sequência de drenagem, limite de R$10.000) vieram de varredura completa do
arquivo (grep/contagem exata sobre as 6.362.620 linhas). O item 7
(nameDest / event stream) foi levantado inicialmente por **amostragem**
(24 a 46 contas), porque o sandbox Linux onde eu rodaria pandas caiu (falha
de hypervisor) durante a sessão original — **José rodou `analisar_namedest.py`
localmente depois e os números reais substituíram as estimativas** (marcado
como `[CONFIRMADO]` abaixo). Uma das estimativas por amostra estava errada
(seção 7.3) — fica registrado como lembrete de que amostra pequena é
hipótese, não conclusão.

## 1. Shape e tipos

- 6.362.620 linhas × 11 colunas.
- `step` (int, unidade de tempo/hora), `type` (categórica), `amount` (float),
  `nameOrig`/`nameDest` (string), `oldbalanceOrg`/`newbalanceOrig`,
  `oldbalanceDest`/`newbalanceDest` (float), `isFraud`/`isFlaggedFraud`
  (binários).
- **Não existe coluna `clienteID`** — diferente do que o script-desafio.md
  presume. Ver seção 7.

## 2. Distribuição dos tipos de transação

| Tipo | Contagem | % |
|---|---|---|
| CASH_OUT | 2.237.500 | 35,17% |
| PAYMENT | 2.151.495 | 33,81% |
| CASH_IN | 1.399.284 | 21,99% |
| TRANSFER | 532.909 | 8,38% |
| DEBIT | 41.432 | 0,65% |

## 3. Desbalanceamento fraude vs legítima

- Legítimas: 6.354.407 (99,871%) · Fraude: 8.213 (0,129%)
- Razão aproximada: 1 fraude para cada 774 transações legítimas.

## 4. Fraude por tipo de transação

| Tipo | Fraudes | % das fraudes |
|---|---|---|
| CASH_OUT | 4.116 | 50,1% |
| TRANSFER | 4.097 | 49,9% |
| PAYMENT / CASH_IN / DEBIT | 0 | 0% |

100% da fraude rotulada está em CASH_OUT e TRANSFER.

## 5. Sequência de eventos de fraude (padrão de drenagem)

Cada fraude no PaySim é sintetizada como um par **TRANSFER → CASH_OUT** no
mesmo `step` (drena o saldo total da origem, depois saca):

```
step=1  TRANSFER   181.00   C1305486145  saldo 181.00→0.00   dest C553264065
step=1  CASH_OUT   181.00   C840083671   saldo 181.00→0.00   dest C38997010
```

`nameOrig` (quem envia) **praticamente nunca se repete** no dataset inteiro —
cada conta de origem aparece uma única vez. Não há histórico sequencial de
cliente-origem para explorar.

## 6. Padrão "card testing" vs limite de R$10.000

- Das 8.213 fraudes em TRANSFER/CASH_OUT, só 278 (3,4%) têm valor abaixo de
  R$10.000 — a maioria da fraude simulada já é de valor alto.
- `isFlaggedFraud` (regra nativa por valor alto) capturou 16 das 8.213
  fraudes reais — 0,19% de recall. Equivalente ao modelo atual da operadora.
- Conclusão: bloqueio por valor absoluto tem recall baixíssimo contra fraude
  estruturada; e ainda bloqueia clientes legítimos de valor alto (ver seção 7).

## 7. nameDest como âncora do event stream — decisão de arquitetura

**Problema identificado**: o script-desafio.md descreve o PaySim como tendo
`clienteID` que permite montar sequência de eventos por cliente. Isso não é
verdade para `nameOrig` (ver seção 5). Decisão tomada com José: usar
**`nameDest`** (conta que recebe) como a entidade da sequência. Muda a
narrativa de negócio: em vez de "detectar se o cliente que envia é
fraudador", o modelo passa a responder **"essa conta de destino está
recebendo um padrão de transações que precede/indica atividade de conta
laranja (mula)?"**.

### 7.1 Quantas contas nameDest têm mais de 1 transação? Distribuição? [CONFIRMADO — censo completo via `analisar_namedest.py`]

- Total de contas `nameDest` únicas (TRANSFER/CASH_OUT): **509.565**
- Com mais de 1 transação: **381.234 (74,82%)**
- Ou seja, ~25% das contas de destino recebem só 1 transação — a estimativa
  anterior por amostra (n=24, nenhuma com count=1) estava **enviesada**; o
  censo completo corrige isso.
- Percentis: mediana **3**, p90 **13**, p99 **28**.

### 7.2 Tamanho médio da sequência por nameDest [CONFIRMADO]

Mediana real é **3 transações por conta** — bem menor que a estimativa
inicial por amostra (~17), que estava enviesada por pegar contas com
atividade acima da média. A distribuição tem cauda longa (p99=28).

**Decisão aprovada: janela = 13** (p90 — cobre 90% das contas sem truncar).
Justificativa: mediana de 3 seria curta demais para o transformer aprender
padrão sequencial; a VRAM da RTX 5060 (~8,5GB) limita o quanto dá para
aumentar a janela sem estourar memória; p90=13 é o equilíbrio entre
cobertura de contas e custo computacional. Contas com menos de 13 eventos
recebem padding no início (ver `montar_event_stream.py`).

### 7.3 Contas nameDest associadas a fraude têm mais transações que as legítimas? [CONFIRMADO — contradiz a estimativa anterior]

Censo completo:

| | Média | Mediana |
|---|---|---|
| Contas com fraude | 5,65 | 3 |
| Contas sem fraude | 5,43 | 3 |

**Volume não separa as classes.** A diferença é desprezível (5,65 vs 5,43).
Isso **contradiz** a estimativa anterior por amostra pequena (n=11), que
sugeria uma diferença maior (~29 vs ~17) — aquela amostra não era
representativa. Fica como aprendizado: conclusões de EDA por amostragem
pequena precisam ser tratadas como hipótese, não fato, até confirmação por
censo completo.

**Implicação direta para a arquitetura**: como volume não é um
discriminador, features de contagem/velocidade (`n_tx_anteriores`,
`soma_valor_anterior` etc., ver `preprocessing/preprocess_pipeline.py`) têm
valor limitado sozinhas. **O transformer precisa aprender o padrão da
sequência (tipo, valor, espaçamento no tempo, saldo), não só o volume** —
reforça a escolha de uma arquitetura de atenção sobre features de contagem
simples.

### 7.3.1 Contas mistas (recebem fraude E legítimo) [novo, do censo completo]

**5.062 contas (0,99%)** recebem tanto transações legítimas quanto pelo
menos uma fraudulenta. Isso confirma que o **label por posição/transação é
a escolha certa** (já aprovada) — um label estático por conta ("essa conta é
mula: sim/não") classificaria errado todas as transações legítimas dessas
~5 mil contas mistas. Com label por posição, cada transação da conta mista
é rotulada individualmente e corretamente.

### 7.4 Estrutura do event stream — o que vira token?

Cada **token = uma transação recebida por aquela conta de destino**
(`nameDest` fixo, uma linha do dataset = um evento). Features sugeridas por
token:

- `delta_step` — steps desde o evento anterior da mesma conta (não o step
  absoluto; posicional encoding deve codificar tempo relativo, não hora do
  dia sozinha)
- `type` (embedding categórico: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- `amount` (normalizado, + flag `amount < limite_atual` e
  `amount_pct_do_limite`)
- `oldbalanceDest` / `newbalanceDest` (saldo da própria conta destino antes/
  depois — captura se a conta está "acumulando e depois zerando", assinatura
  de conta mula)
- `is_amount_round` (heurística: valor "redondo" ou testado — proxy fraco de
  card testing, já que a conta de origem raramente se repete)

Sequência = últimas **13** transações recebidas por essa conta destino
(janela confirmada na seção 7.2), ordenadas por `step` crescente, com padding
no início para contas com menos histórico. Token especial `[CLS]` no fim da
sequência para classificação (saída sigmoid). Implementado em
`preprocessing/montar_event_stream.py` e `modelo/transformer_sequencial.py`.

### 7.5 Como definir o label da conta destino? [DECIDIDO: label por posição]

**Decidido: label por posição/transação**, não por conta. Para a transação
`k` recebida por uma conta, o rótulo é o `isFraud` daquela transação
específica; o contexto de entrada usa só as transações `0..k-1` da mesma
conta (sem vazamento). Implementado em `montar_event_stream.py`.

Motivo confirmado pelo censo (seção 7.3.1): **5.062 contas (0,99%) são
mistas** (recebem legítimas e fraude). Um label estático por conta ("essa
conta é mula: sim/não") classificaria errado todas as transações legítimas
dessas contas mistas. Label por posição resolve isso e também está mais
alinhado com a meta de bloqueio em tempo real (avaliar a cada evento novo,
olhando para trás).

## Recomendações para as próximas fases

- ~~Rodar `eda/analisar_namedest.py` no dataset completo~~ — feito. Números
  da seção 7 confirmados por censo completo (509.565 contas).
- `preprocessing/montar_event_stream.py` já implementa o agrupamento por
  `nameDest`, ordenação por `step`, janela=13 e o corte "sem vazamento".
- `preprocessing/preprocess_pipeline.py` calcula features de velocidade
  complementares (`n_tx_anteriores`, `soma_valor_janela_curta`,
  `sinal_estruturacao` etc.) — mas seção 7.3 mostra que volume sozinho não
  discrimina fraude, então essas features têm valor limitado como sinal
  isolado; o transformer precisa aprender o padrão da sequência.
- Métrica de sucesso definida como por transação (label por posição) —
  ver seção 7.5. Validar threshold de bloqueio com o cliente na FASE 7.
