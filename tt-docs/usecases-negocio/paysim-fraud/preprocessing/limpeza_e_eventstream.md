# Limpeza e event stream — decisões e justificativas

## Escopo dos dados

Só `TRANSFER` e `CASH_OUT` entram no pipeline de modelagem — são os únicos
tipos com fraude rotulada (ver `EDA_paysim.md` item 4). `PAYMENT`, `CASH_IN`
e `DEBIT` ficam de fora do treino do modelo sequencial; se a operadora quiser
cobertura desses tipos, precisa de rótulos adicionais (não existem no PaySim).

## Duplicatas

`preprocess_pipeline.py` remove duplicatas exatas de linha antes de qualquer
outra etapa. No PaySim isso tende a ser raro (dataset sintético), mas é
checado e reportado.

## Entidade da sequência: nameDest, não nameOrig

Ver `EDA_paysim.md` seção 7 para o achado completo. Resumo: o dataset não
tem `clienteID`; `nameOrig` não se repete; `nameDest` concentra múltiplas
transações e vira a âncora do event stream. A narrativa de negócio muda de
"detectar cliente fraudador" para "detectar conta de destino com padrão de
conta mula".

## Corte sem vazamento

`montar_event_stream.py` monta, para cada posição `k` na sequência de uma
conta, um contexto usando **só** os eventos `0..k-1`. O evento `k` (cujo
`isFraud` vira o rótulo) nunca entra no contexto de entrada do modelo. Isso é
necessário porque a meta do projeto é "detectar antes da fraude principal" —
um modelo que vê o próprio evento rotulado no contexto está trapaceando.

## Split por conta, não por transação

Se o split fosse por transação, a mesma conta `nameDest` poderia aparecer em
treino e teste simultaneamente (transações diferentes da mesma conta em
splits diferentes) — vazamento de identidade. `preprocess_pipeline.py` faz
o split embaralhando `nameDest` únicos e distribuindo 70/15/15, garantindo
que todas as transações de uma conta fiquem no mesmo split.

## Normalização

Z-score calculado **apenas no split de treino** e aplicado a validação/teste
com as mesmas médias/desvios (evita vazamento estatístico). `preprocessing_meta.json`
agora tem duas chaves de topo para não se sobrescreverem (`atualizar_meta_json()`
faz merge, não overwrite):

- `tabular` — scaler das features agregadas de `preprocess_pipeline.py`
  (`amount`, saldos de origem, `erroBalance*`, features de velocidade).
- `event_stream` — scaler de `montar_event_stream.py`, usado para normalizar
  os tensores que o transformer realmente consome (`amount`, `delta_step`,
  `oldbalanceDest`, `newbalanceDest` dentro de cada token da sequência).

**Bug corrigido**: até a versão anterior, `montar_event_stream.py` montava
o event stream com esses 4 canais em escala bruta (valores na casa de
dezenas de milhões, ver EDA), sem normalizar. Isso não aparecia no scaler de
`preprocess_pipeline.py` porque aquele scaler nunca alimentava o
`event_stream.npz` — os dois pipelines eram independentes. O sintoma foi
`logit` não-finito já no forward pass (diagnosticado com
`diagnosticar_primeiro_batch()` em `transformer_sequencial.py`): um
`nn.Linear` recebendo `oldbalanceDest`/`newbalanceDest` na casa de milhões
sem normalização estoura facilmente. Agora `montar_event_stream.py` calcula
o z-score desses 4 canais **só nas contas do split de treino** (usando a
mesma `split_por_conta()` e a mesma seed de `preprocess_pipeline.py`) e
aplica nos três splits antes de salvar o `.npz`.

Importante: o `.npz` agora também salva um array `split` (`"treino"`/`"val"`/
`"teste"`, por exemplo), mas `transformer_sequencial.py` ainda faz o próprio
`random_split` 85/15 ignorando esse campo — ou seja, o vazamento de conta
entre treino/validação do transformer **ainda não foi corrigido**, só a
normalização. Ajustar isso é o próximo passo pendente antes de confiar
plenamente nas métricas de validação.

## Features de velocidade (complemento tabular, decisão confirmada com José)

`preprocess_pipeline.py` calcula `n_tx_anteriores`, `soma_valor_anterior`,
`max_valor_anterior`, `soma_valor_janela_curta`, `n_tx_janela_curta` e o flag
`sinal_estruturacao` — tudo por `nameDest`, respeitando o mesmo corte sem
vazamento do event stream (só olha para transações antes da linha atual).
Isso substitui a ideia original de `build_velocity_features()` (que agrupava
por `nameOrig`, obsoleta desde a decisão da seção 7 do `EDA_paysim.md`).

Importante: essas features são agregados tabulares, não substituem o event
stream bruto que o transformer consome (`montar_event_stream.py`). Servem
para dois fins: (1) alimentar um baseline mais simples (ex.: gradient
boosting) para comparar com o AUPRC do transformer, e (2) dar ao próprio
transformer, se quisermos, features adicionais por token além das 5
originais — decisão em aberto, não implementada ainda no
`transformer_sequencial.py`.

`sinal_estruturacao` é a tradução direta do "card testing" para conta de
destino: soma recente de entradas já passou do limite de R$10.000, mas
nenhuma transação individual anterior passou sozinha. `preprocess_pipeline.py`
imprime quantas transações acendem esse sinal e quantas dessas são fraude de
verdade — útil para validar se o sinal tem alguma correlação com `isFraud`
antes de depender dele.

## Class weight

Com desbalanceamento de ~0,13% (ver EDA item 3), `preprocessing_meta.json`
inclui uma sugestão de peso de classe (`class_weight_sugerido`) para usar na
loss do transformer — ponto de partida, não um valor definitivo; ajustar
conforme o threshold de negócio decidido com o cliente (FASE 1/7 do
script-desafio.md).
