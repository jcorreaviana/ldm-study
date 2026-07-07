# Pseudocódigos — Machine Learning do Zero ao Transformer
> Representação em linguagem natural dos principais algoritmos implementados.
> Objetivo: fixar a lógica de cada solução sem depender da sintaxe Python.

---

## 1. Regressão Linear do Zero

**Problema:** dado um dataset de (x, y), encontrar a e b que minimizam o erro.

```
ALGORITMO: Regressão Linear

ENTRADA: dataset [(x1,y1), ..., (xn,yn)], learning_rate, épocas
SAÍDA:   parâmetros a e b otimizados

INÍCIO
  média_x ← soma(x) / n
  
  PARA cada x:
    x_normalizado ← x - média_x        ← z-score simplificado

  inicializa a ← 1.0, b ← 1.0

  PARA cada época de 1 até épocas:

    // FORWARD PASS
    PARA cada (x, y) no dataset:
      y_prev ← a × x_normalizado + b   ← aplica o modelo
      erro   ← (y_prev - y)²           ← erro quadrático

    perda ← média dos erros            ← MSE

    // BACKWARD PASS (regra da cadeia)
    dL_da ← (1/n) × soma(2 × e × x_norm)  ← ∂L/∂a
    dL_db ← (1/n) × soma(2 × e)            ← ∂L/∂b

    // ATUALIZAÇÃO (gradiente descendente)
    a ← a - learning_rate × dL_da
    b ← b - learning_rate × dL_db

  RETORNA a, b
FIM
```

---

## 2. Regressão Linear com Early Stopping e R²

**Adiciona:** divisão treino/validação/teste, early stopping, R².

```
ALGORITMO: Regressão Linear v2

ENTRADA: dataset, learning_rate, patience
SAÍDA:   a, b otimizados, R² no conjunto de teste

INÍCIO
  divide dataset em treino (70%), validação (15%), teste (15%)
  calcula média e desvio de cada feature usando APENAS o treino

  inicializa a ← 1.0, b ← 1.0
  melhor_val ← infinito
  contador_patience ← 0

  ENQUANTO contador_patience < patience:

    // treina com dados de treino
    dL_da, dL_db ← gradientes(treino, a, b)
    a ← a - lr × dL_da
    b ← b - lr × dL_db

    // avalia no conjunto de validação
    perda_val ← MSE(validacao, a, b)

    SE perda_val melhorou (< melhor_val - min_delta):
      melhor_val ← perda_val
      contador_patience ← 0        ← reseta
    SENÃO:
      contador_patience ← contador_patience + 1  ← incrementa

  // avaliação final — usado UMA só vez
  R² ← 1 - SS_residual / SS_total   calculado em teste

  RETORNA a, b, R²
FIM
```

---

## 3. Regressão com Múltiplas Features

**Adiciona:** vetor de pesos W, z-score completo, produto escalar.

```
ALGORITMO: Regressão Múltipla

ENTRADA: dataset [([x1,x2,...], y)], learning_rate, épocas
SAÍDA:   vetor W e viés b otimizados

INÍCIO
  // normalização z-score por feature
  PARA cada feature j:
    media[j]  ← média de todos os x[j] no dataset
    desvio[j] ← desvio padrão de todos os x[j] no dataset

  FUNÇÃO normalizar(X):
    RETORNA [(X[j] - media[j]) / desvio[j] para cada j]

  // modelo: produto escalar
  FUNÇÃO modelo(X_norm, W, b):
    RETORNA soma(W[j] × X_norm[j] para cada j) + b

  inicializa W ← [1.0, 1.0, ...], b ← 1.0

  PARA cada época:
    PARA cada (X, y) no dataset:
      X_norm ← normalizar(X)
      y_prev ← modelo(X_norm, W, b)
      e      ← y_prev - y

      // gradiente por feature
      PARA cada j:
        dL_dW[j] ← dL_dW[j] + 2 × e × X_norm[j]
      dL_db ← dL_db + 2 × e

    // atualiza cada peso
    PARA cada j:
      W[j] ← W[j] - lr × dL_dW[j] / n
    b ← b - lr × dL_db / n

  RETORNA W, b
FIM
```

---

## 4. Rede Neural com Múltiplas Camadas

**Adiciona:** camadas ocultas, leaky relu, He initialization, gradient clipping.

```
ALGORITMO: Rede Neural Multicamadas

ENTRADA: dataset, n_camadas, n_ocultos, learning_rate
SAÍDA:   pesos de todas as camadas otimizados

INÍCIO
  // inicialização He — evita vanishing/exploding gradient
  PARA cada camada l:
    escala ← raiz(2 / n_entradas_camada_l)
    camadas_W[l] ← matriz aleatória com média 0 e desvio escala
    camadas_b[l] ← zeros

  FUNÇÃO forward(X):
    entrada ← X
    PARA cada camada l:
      PARA cada neurônio i:
        z[i] ← produto_escalar(entrada, camadas_W[l][:][i]) + camadas_b[l][i]
        h[i] ← leaky_relu(z[i])      ← ativação
      guarda h e z para uso no backward
      entrada ← h                     ← saída vira entrada da próxima
    RETORNA saída, histórico de h e z

  FUNÇÃO backward(X, y, todas_h, todos_z):
    e ← saída - y                     ← erro na saída
    delta ← gradiente inicial

    // percorre de trás para frente
    PARA cada camada l de N até 1:
      drelu ← 1 se z > 0, senão alpha  ← derivada do leaky relu
      gradiente ← delta × drelu

      atualiza dL_dW[l] e dL_db[l]

      // propaga gradiente para camada anterior
      delta ← soma(gradiente × camadas_W[l])

    RETORNA gradientes de todas as camadas

  PARA cada época:
    y_prev, todas_h, todos_z ← forward(X)
    gradientes ← backward(X, y, todas_h, todos_z)

    // atualiza com gradient clipping
    PARA cada peso w:
      grad ← clip(gradiente[w], -max_grad, +max_grad)
      w ← w - lr × grad

  RETORNA camadas_W, camadas_b
FIM
```

---

## 5. Mecanismo de Atenção

**Problema:** calcular relevância entre tokens de uma sequência.

```
ALGORITMO: Self-Attention

ENTRADA: tokens [n × d_model], matrizes Wq, Wk, Wv
SAÍDA:   tokens enriquecidos com contexto [n × d_k]

INÍCIO
  // etapa 1: criar Q, K, V
  Q ← tokens × Wq          ← [n × d_k]
  K ← tokens × Wk          ← [n × d_k]
  V ← tokens × Wv          ← [n × d_k]

  // etapa 2: calcular scores de relevância
  scores ← Q × transposta(K)       ← [n × n]
  scores ← scores / raiz(d_k)      ← escala para estabilidade

  // etapa 3: softmax — transforma em probabilidades
  PARA cada linha i de scores:
    atenção[i] ← softmax(scores[i])   ← soma = 1.0

  // etapa 4: saída ponderada
  saída ← atenção × V               ← [n × d_k]

  RETORNA saída
  // cada token agora "sabe" o que aconteceu nos outros
FIM
```

---

## 6. Pipeline Completo: Pré-processamento + Atenção

**Problema:** transformar eventos brutos em representação para classificação de fraude.

```
ALGORITMO: Pipeline Fraude

ENTRADA: sequência de eventos brutos [{valor, hora, país, device}]
SAÍDA:   score de fraude entre 0 e 1

INÍCIO
  // pré-processamento
  PARA cada evento:
    z_valor  ← (valor - media_valor) / desvio_valor   ← z-score
    z_hora   ← (hora  - media_hora)  / desvio_hora    ← z-score
    pais_vec ← one_hot(pais, ['brasil','eua','europa','asia','outro'])
    device_b ← 1 se conhecido, 0 se desconhecido      ← binário

    token ← [z_valor, z_hora] + pais_vec + [device_b]  ← vetor 8D

  // mecanismo de atenção
  Q, K, V ← tokens × Wq, tokens × Wk, tokens × Wv
  scores  ← softmax(Q × Kᵀ / raiz(d_k))
  saída   ← scores × V

  // classificação
  z     ← produto_escalar(ultimo_token_saída, W_class)
  score ← sigmoid(z)    ← probabilidade de fraude 0-1

  RETORNA score
FIM
```

---

## 7. Bloco Transformer Completo

**Problema:** processar sequência de tokens com atenção, feed-forward e residual.

```
ALGORITMO: Bloco Transformer

ENTRADA: tokens X [n × d_model]
SAÍDA:   tokens enriquecidos [n × d_model]  ← mesma dimensão!

INÍCIO
  // sub-bloco 1: atenção
  atencao_saida ← multi_head_attention(X)
  residual_1    ← X + atencao_saida          ← residual connection
  norm_1        ← layer_norm(residual_1)      ← z-score nas ativações

  // sub-bloco 2: feed-forward
  PARA cada token em norm_1:
    z1    ← token × W1 + b1                  ← expande: d_model → d_ff
    h1    ← relu(z1)                          ← ativação
    z2    ← h1 × W2 + b2                      ← comprime: d_ff → d_model

  residual_2 ← norm_1 + feed_forward_saida   ← residual connection
  norm_2     ← layer_norm(residual_2)         ← z-score nas ativações

  RETORNA norm_2

// empilha N blocos em sequência
PARA cada bloco de 1 até N:
  X ← bloco_transformer(X)

// classificação final
score ← sigmoid(X[-1] × W_class)   ← usa último token
FIM
```

---

## 8. Transformer com Treinamento (PyTorch)

**Adiciona:** backward automático, AdamW, BCE loss, convergência real.

```
ALGORITMO: Transformer Treinado

ENTRADA: dataset rotulado [(sequência, label)], épocas
SAÍDA:   modelo treinado capaz de classificar fraude

INÍCIO
  // arquitetura
  modelo ← N blocos transformer + classificador sigmoid

  // treinamento
  otimizador ← AdamW(parâmetros do modelo, lr=0.001)
  criterio   ← Binary Cross-Entropy

  PARA cada época:
    // forward pass
    scores ← modelo(X)                    ← passa pelos N blocos

    // calcula perda
    perda ← BCE(scores, labels)
    // perda alta quando modelo erra com confiança
    // perda = -[y × log(score) + (1-y) × log(1-score)]

    // backward pass (automático no PyTorch)
    perda.backward()
    // calcula ∂perda/∂w para TODOS os pesos automaticamente
    // inclui Wq, Wk, Wv, W1, W2 de todos os blocos

    // atualiza pesos (AdamW)
    otimizador.step()
    otimizador.zero_grad()

    // convergência: perda para de cair → modelo aprendeu

  RETORNA modelo treinado
FIM
```

---

## 9. LoRA — Fine-tuning Eficiente

**Problema:** adaptar modelo pré-treinado para novo domínio sem retreinar tudo.

```
ALGORITMO: LoRA Fine-tuning

ENTRADA: modelo pré-treinado (W_original), dataset do novo domínio, rank
SAÍDA:   matrizes A e B que representam a adaptação

INÍCIO
  // congela todos os pesos originais
  W_original ← CONGELADO (não será atualizado)

  // inicializa matrizes de baixo rank
  A ← matriz aleatória [d_model × rank]    ← pequena
  B ← zeros            [rank × d_model]    ← começa em zero

  // durante o forward pass
  W_efetivo ← W_original + A × B           ← adaptação somada

  // treinamento — só A e B recebem gradientes
  PARA cada época:
    saída ← modelo_com_lora(X, W_original, A, B)
    perda ← criterio(saída, labels)
    perda.backward()
    // apenas ∂perda/∂A e ∂perda/∂B são calculados
    // W_original permanece congelado

    atualiza A e B

  // resultado: W_original intacto + (A × B) = adaptação aprendida
  RETORNA A, B
FIM
```

---

## Resumo Visual dos Algoritmos

```
REGRESSÃO LINEAR
  dados → normalizar → f(x)=ax+b → MSE → ∂L/∂a, ∂L/∂b → atualiza a,b

REDE NEURAL
  dados → [camada+relu] × N → saída → BCE/MSE → backward → atualiza todos W

MECANISMO DE ATENÇÃO
  tokens → Q,K,V → Q·Kᵀ/√d → softmax → ×V → tokens enriquecidos

BLOCO TRANSFORMER
  tokens → [atenção + residual + layernorm] → [FF + residual + layernorm] → tokens

TRANSFORMER COMPLETO
  pré-proc → embedding → N×bloco → sigmoid → score
           ↑ forward                        ↓ BCE loss
           └──────────── backward ──────────┘

LoRA
  W_novo = W_original(congelado) + A×B(treinável)
```