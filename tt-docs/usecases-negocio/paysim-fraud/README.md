# PaySim — detecção de conta laranja com event stream + transformer sequencial

Segundo projeto prático (evolução do credit card fraud tabular). Ver
`script-desafio.md` para o enunciado original do curso.

## Achado que muda o projeto: não existe clienteID

O enunciado (`script-desafio.md`) presume uma coluna `clienteID` que permite
montar sequência de eventos por cliente. O dataset real não tem isso — só
`nameOrig` (quem envia) e `nameDest` (quem recebe). Na EDA confirmei que
`nameOrig` praticamente nunca se repete (cada conta de origem transaciona uma
vez só), então não há histórico sequencial de "cliente que envia" para
explorar.

Decisão tomada com José: usar **`nameDest`** como âncora do event stream. A
pergunta de negócio muda de "esse cliente que está enviando é fraudador?"
para **"essa conta de destino está recebendo um padrão de transações que
indica que é uma conta laranja (mula)?"**. Justificativa completa e números
em `eda/EDA_paysim.md` seção 7.

## Decisões de configuração confirmadas

- **Label: por posição/transação**, não por conta (5.062 contas — 0,99% —
  recebem legítimas e fraude; label por conta classificaria errado as
  transações legítimas dessas contas mistas).
- **Janela de contexto: 13 eventos.** Censo completo (`analisar_namedest.py`)
  mostrou p90 = 13 transações por `nameDest` (mediana real é só 3 — curta
  demais para o transformer aprender padrão sequencial). VRAM da RTX 5060
  (~8,5GB) também limita o quanto dá para aumentar a janela.
- **Volume não separa fraude de legítimo** (média 5,65 vs 5,43 transações
  por conta) — o transformer precisa aprender o padrão da sequência (tipo,
  valor, espaçamento, saldo), não a contagem de eventos.
- Registro completo em `avaliacao/resultado_final.md`.

## Bug corrigido: logit não-finito no forward (não na loss)

Depois de corrigir pos_weight/autocast/grad clipping, o treino ainda
retornava perda não-finita — mesmo com FocalLoss. Diagnóstico (prints
adicionados em `transformer_sequencial.py`) confirmou: o problema era
**anterior à loss**, no forward pass. Causa: `amount`, `oldbalanceDest` e
`newbalanceDest` entravam sem normalização (valores na casa de milhões, ver
EDA) direto num `nn.Linear`, estourando. Corrigido em
`preprocessing/montar_event_stream.py`: z-score desses três canais + 
`delta_step`, calculado **só nas contas do split de treino** (mesma
`split_por_conta()` de `preprocess_pipeline.py`, mesma seed), aplicado nos
três splits antes de salvar `event_stream.npz`. Estatísticas salvas em
`preprocessing_meta.json` (chave `event_stream`) para uso na inferência em
produção. Detalhes em `preprocessing/limpeza_e_eventstream.md`.

**Corrigido também**: `transformer_sequencial.py` usava `random_split()` por
exemplo (não por conta) para treino/validação, o que podia deixar a mesma
conta `nameDest` em treino e validação ao mesmo tempo. Agora usa o array
`split` do `.npz` (por conta) via `carregar_event_stream()` — `EventStreamDataset`
recebe arrays já fatiados em vez de carregar o arquivo sozinho.
`avaliacao/avaliar_modelo.py` foi ajustado junto para avaliar só no split
`"teste"` (antes avaliava tudo misturado).

## Diagnóstico: transformer não discrimina nada (diferença de scores 0.0001)

Depois de corrigir GPU/DataLoader (`num_workers`, `pin_memory`, batch maior)
e subir o lr pra 1e-3, o AUPRC continuou estagnado perto do piso e a
diferença de score médio entre fraude e legítima na validação veio 0.0001 —
o modelo não aprendeu nada discriminativo.

Causa raiz identificada: o event stream (ancorado em `nameDest`, ver acima)
**não tem acesso a `oldbalanceOrg`/`newbalanceOrig`** (saldo de quem envia).
É exatamente aí que mora o sinal clássico de fraude do PaySim — um
`TRANSFER` que drena ~100% do saldo do remetente, seguido de `CASH_OUT`
imediato. `erroBalanceOrig` (já calculada em `preprocess_pipeline.py`) capta
isso diretamente, mas vive só no pipeline tabular, nunca chega ao
transformer. `montar_event_stream.py` só carrega `step, type, amount,
nameDest, oldbalanceDest, newbalanceDest, isFraud` do CSV — nunca
`nameOrig`/`oldbalanceOrg`/`newbalanceOrig`.

Para isolar "falta sinal no pipeline do transformer" de "problema de
arquitetura/otimização", criei `modelo/baseline_tabular.py`: XGBoost (ou
regressão logística, se xgboost não estiver instalado) nas features
tabulares completas do `preprocess_pipeline.py`, incluindo `erroBalanceOrig`.
Resultado ainda não rodado — ver `avaliacao/resultado_baseline.md`.

## Estrutura

```
eda/                  análise exploratória (shape, distribuições, event stream)
preprocessing/        limpeza, feature engineering, montagem do event stream
modelo/                transformer sequencial (PyTorch) + positional encoding
avaliacao/             curva PR, matriz de confusão, relatório final
gpu_monitoring/        logger de GPU (pynvml) + integração wandb
```

Dataset (`dataset.csv`) fica na raiz do projeto mas **não é versionado** —
só os scripts. Todo script aponta para ele via `--path`.

## Ordem de execução

```bash
# 1. EDA (gera os PNGs em eda/eda_out/)
python eda/paysim_eda.py --path dataset.csv
python eda/analisar_namedest.py --path dataset.csv    # já rodado - números confirmados na seção 7 do EDA_paysim.md

# 2. Pré-processamento
python preprocessing/preprocess_pipeline.py --path dataset.csv --output-dir ../../paysim_data
python preprocessing/montar_event_stream.py --path dataset.csv --output-dir ../../paysim_data --janela 13

# 3. Baseline tabular (rodar ANTES de continuar ajustando o transformer -
#    ver "Diagnóstico" acima)
python modelo/baseline_tabular.py --path dataset.csv

# 4. Treino do transformer (RTX local, monitorar com nvitop em outro terminal)
python modelo/transformer_sequencial.py --data ../../paysim_data/event_stream.npz --lr 1e-3

# 5. Avaliação
python avaliacao/avaliar_modelo.py --data ../../paysim_data/event_stream.npz --checkpoint modelo/checkpoint_melhor_modelo.pt
```

`../../paysim_data/` (`tt-docs/paysim_data/`, tensores derivados fora deste
diretório) também não deve ser versionado.

## O que ainda não existe neste repositório (e por quê)

Estes arquivos aparecem na estrutura mas não foram gerados agora porque
dependem de execução real (treino/GPU), que não fiz — eu geraria resultados
fabricados, o que seria enganoso em um projeto de detecção de fraude:

- `modelo/checkpoint_melhor_modelo.pt` — gerado ao rodar o treino de verdade.
- `avaliacao/curva_pr.png`, `avaliacao/matriz_confusao.png` — gerados por
  `avaliacao/avaliar_modelo.py` a partir de um checkpoint real.
- `avaliacao/resultado_final.md`, `avaliacao/resultado_baseline.md` e
  `avaliacao/gpu_report.md` — estão como templates, sobrescritos com números
  reais após rodar `baseline_tabular.py` / treino+avaliação do transformer.
- `preprocessing/preprocessing_meta.json` — a chave `tabular` já tem valores
  reais (José rodou `preprocess_pipeline.py` localmente); a chave
  `event_stream` ainda é template, preenchida quando `montar_event_stream.py`
  rodar com a correção de z-score.

A EDA (itens 1–6) usou varredura completa das 6.362.620 linhas. A análise de
`nameDest` (item 7) foi inicialmente feita por amostragem (sandbox de
execução caiu durante a sessão); José rodou `eda/analisar_namedest.py`
localmente depois e os números reais (censo completo de 509.565 contas)
substituíram as estimativas — ver `eda/EDA_paysim.md` seção 7, marcado
`[CONFIRMADO]`.

## Metas (script-desafio.md)

```
técnica:  AUPRC > 0.80 (baseline tabular anterior: 0.705)
negócio:  detectar padrão de conta laranja antes da fraude principal
gpu:      treino completo em < 10 min na RTX local
```
