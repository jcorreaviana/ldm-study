# Índice de Estudos — Machine Learning do Zero ao Transformer
> Guia de estudo construído de forma incremental, partindo de conceitos matemáticos fundamentais até a arquitetura de transformers e LDMs.
> Cada módulo pressupõe o anterior. Todos os conceitos são implementados do zero antes de usar bibliotecas.

---

## Legenda

```
✅  implementado do zero — criação de código com Claude, revisão de conceitos fundamentais no caderno
📄  conceitos estudados via material de referência NeoSpace
⬜  pendente
```

---

## MÓDULO 1 — Fundamentos Matemáticos ✅

### 1.1 Funções e Domínio/Imagem ✅
- O que é uma função f: X → Y
- Domínio (conjunto de entradas) e imagem (conjunto de saídas)
- Funções compostas f(g(x))
- Representação no plano cartesiano
- Quadrantes e interpretação geométrica

### 1.2 Derivadas ✅
- O conceito de inclinação de uma curva
- Derivada como limite da inclinação
- Derivada de xⁿ → nxⁿ⁻¹ (padrão do dobro)
- Tipos de pontos: máximo, mínimo, inflexão
- Derivada = 0 no máximo e mínimo

### 1.3 Regra da Cadeia ✅
- Derivada de funções compostas
- dL/dx = (dL/du) · (du/dx)
- Aplicação em múltiplos níveis

### 1.4 Derivadas Parciais ✅
- Derivada em relação a um parâmetro mantendo os outros fixos
- Notação ∂L/∂a e ∂L/∂b
- Gradiente como vetor de derivadas parciais

### 1.5 Álgebra Linear ✅
- Vetores e espaços Rⁿ
- Produto escalar (dot product): W · X = Σ wᵢxᵢ
- Multiplicação de matrizes e dimensões
- Regra de compatibilidade: colunas de A = linhas de B
- Vetor linha vs vetor coluna
- Transposta de matriz

---

## MÓDULO 2 — Regressão Linear ✅

### 2.1 O Modelo ✅
- Equação da reta: f(x) = a·x + b
- Parâmetros treináveis (a, b) vs dados (x, y)
- Domínio e imagem no contexto de ML
- Quando usar regressão linear vs modelo não-linear

### 2.2 Normalização ✅
- Por que normalizar: equilibrar escalas entre features
- Subtração da média: x' = x - μ
- Z-score: x' = (x - μ) / σ
- Desvio padrão: σ = √(média dos quadrados das diferenças)
- Quando usar cada tipo
- Normalização baseada apenas nos dados de treino

### 2.3 Função de Perda — MSE ✅
- Erro quadrático: (y_prev - y_real)²
- Por que elevar ao quadrado (sempre positivo, diferenciável)
- MSE: L = (1/n) · Σ (f(xᵢ) - yᵢ)²
- Objetivo: minimizar L

### 2.4 Gradiente Descendente ✅
- Derivadas parciais ∂L/∂a e ∂L/∂b
- Regra da cadeia aplicada manualmente
- Atualização: a ← a - η · ∂L/∂a
- Learning rate η e seu papel
- Sinal do gradiente determina a direção

### 2.5 Loop de Treinamento ✅
- Forward pass: calcula previsão e perda
- Backward pass: calcula gradientes
- Atualização dos parâmetros
- Épocas e critério de parada

### 2.6 Avaliação do Modelo ✅
- Overfitting vs generalização — ✅ + 📄
- Divisão treino / validação / teste (70/15/15) ✅
- Early Stopping e patience — ✅ + 📄
- R² — coeficiente de determinação ✅
- Gap treino/validação como diagnóstico ✅
- AUROC e AUPRC como métricas de avaliação 📄
- Precision, Recall e F1 Score 📄
- Matriz de confusão (TP, FP, TN, FN) 📄
- Limiar (threshold) e sua relação com Precision/Recall 📄

### 2.7 Regressão Múltipla — Múltiplas Features ✅
- f(X) = W · X + b (produto escalar)
- Vetor de pesos W = [w1, w2, ..., wn]
- Z-score necessário com features de escalas diferentes
- Representação matricial do dataset
- Um gradiente por feature: dL/dW = [dL/dw1, dL/dw2, ...]

### 2.8 Representação de Dados ✅
- Features numéricas contínuas → z-score
- Features binárias → 0 ou 1, sem normalização
- Features categóricas exclusivas → one-hot encoding
- Features categóricas múltiplas → multi-hot encoding

---

## MÓDULO 3 — Redes Neurais ✅

### 3.1 Função de Ativação ✅
- Por que ativação é necessária (colapso linear)
- ReLU: f(x) = max(0, x)
- Leaky ReLU: f(x) = x se x>0, α·x se x≤0
- Derivada do ReLU: 1 se z>0, 0 se z≤0
- Derivada do Leaky ReLU: 1 se z>0, α se z≤0
- Funções de saída: sigmoid 📄, softmax, nenhuma
- Sigmoide: score = 1 / (1 + e⁻ᶻ) → transforma z em probabilidade 0-1 📄

### 3.2 Neurônio Único ✅
- h = relu(W · X + b)
- Forward pass de um neurônio
- Backward pass: drelu como portão do gradiente
- Dying ReLU — neurônio morto
- Leaky ReLU como solução

### 3.3 Camada com Múltiplos Neurônios ✅
- Matriz de pesos W [n_features × n_neurônios]
- W[j][i] = peso da feature j para o neurônio i
- Symmetry breaking — inicialização aleatória
- Contagem de parâmetros: N(F+2) + 1
- h = vetor de saídas [h1, h2, ..., hN]

### 3.4 Rede Neural com Múltiplas Camadas ✅
- Entrada de cada camada = saída da camada anterior
- Forward pass percorre camadas em sequência
- Backward pass percorre de trás para frente
- Contagem de parâmetros com L camadas: N(F+1) + (L-1)·N(N+1) + N+1
- Vanishing gradient em redes profundas
- Exploding gradient — overflow
- He initialization: escala = √(2/n_entradas)
- Gradient clipping: limita gradientes ao máximo

### 3.5 Técnicas de Otimização ✅
- Gradient clipping
- He initialization
- Leaky ReLU vs ReLU
- Learning rate scheduling (warmup, cooldown, cosine annealing)
- AdamW — otimizador adaptativo com weight decay correto
- Grid Search e Random Search para hiperparâmetros
- Early Stopping com patience

---

## MÓDULO 4 — Base do Transformer ✅

### 4.1 Residual Connections ✅
- saída = transformação(entrada) + entrada
- Caminho direto para o gradiente
- Resolve vanishing gradient

### 4.2 Layer Normalization ✅
- z-score aplicado nas ativações: x' = (x - μ) / σ
- Mantém gradientes em escala controlada entre camadas

### 4.3 Bloco Residual ✅
- [Layer Norm → Dense → ReLU] + conexão residual
- Comparativo: rede densa vs rede residual

---

## MÓDULO 5 — Transformer ✅

### 5.1 Embeddings 📄
- Representação de tokens como vetores densos
- Tokens similares ficam próximos no espaço vetorial

### 5.2 Tokenização 📄
- O que é um token (palavra em NLP, evento no LDM)
- Contexto máximo (context window)

### 5.3 Positional Encoding 📄
- Por que posição importa em sequências
- Sem positional encoding o transformer trata sequência como desordenada

### 5.4 Mecanismo de Atenção ✅
- Query, Key, Value (Q, K, V)
- scores = Q · Kᵀ / √d_k
- Softmax: eˣ / Σeˣ → probabilidades que somam 1
- saída = softmax(scores) · V
- Pipeline completo: pré-processamento → token → Q/K/V → atenção → saída enriquecida
- Transposta: linhas viram colunas
- Por que dividir por √d (estabilidade dos gradientes)

### 5.5 Multi-Head Attention ✅
- Múltiplas cabeças em paralelo
- Cada cabeça aprende um aspecto diferente
- A responsabilidade de cada cabeça emerge do treinamento

### 5.6 Arquitetura Completa do Transformer ✅
- Bloco: [Multi-Head Attention + Residual + LayerNorm] + [Feed-Forward + Residual + LayerNorm]
- Feed-forward: d_model → 4×d_model → d_model
- N blocos empilhados — entrada e saída sempre [n_tokens × d_model]
- Sigmoid na saída para classificação binária
- Implementado do zero e com PyTorch (com treinamento completo)

---

## MÓDULO 6 — LLMs e Aplicações ✅

### 6.1 Funções de Perda para Classificação ✅
- Cross-entropy: penaliza erros de confiança exponencialmente
- Binary Cross-Entropy (BCE): L = -[y·log(score) + (1-y)·log(1-score)]
- Categorical Cross-Entropy: usada em LLMs para prever próximo token
- Ciclo: forward → BCE → gradiente → atualiza pesos

### 6.2 Fine-tuning Eficiente ✅
- LoRA: W_novo = W_original (congelado) + A × B (treinável)
- Rank: dimensão interna de A e B — sweet spot 4-16
- QLoRA: comprime W_original de float16 → int4, mantém A×B em float16
- Quando retreinar do zero: pré-treino ruim, drift catastrófico, mudança estrutural

### 6.3 Inferência e Hardware ✅
- VRAM e capacidade de parâmetros
- Quantização (float32, float16, int8, Q4)
- Modelos abertos: Llama, Mistral, Qwen
- Ferramentas: Ollama, Hugging Face, Unsloth

### 6.4 Pipeline Clássico vs LDM 📄
- Feature engineering manual → event stream bruto
- Um modelo por tarefa → modelo unificado
- Arquitetura LDM: 5 blocos
- Ciclo de valor: pre-train → fine-tune → prescriptive

---

## Estrutura do Repositório

```
tt-docs/
│
├── math-examples/
│   └── regressao/
│       ├── regressao_do_zero.py
│       ├── regressao_do_zero_v2.py
│       ├── regressao_multiplas_features.py
│       ├── regressao_visual.py
│       └── comparativo_zero_vs_micrograd.py
│
├── rna/
│   ├── relu_comparativo.py
│   ├── rede_neural_3_neuronios.py
│   ├── rede_neural_leaky_relu.py
│   └── rede_neural_multicamadas.py
│
├── transformer/
│   ├── transformer_vs_densa.py
│   ├── mecanismo_atencao.py
│   ├── pipeline_completo.py
│   ├── transformer_completo.py
│   └── transformer_treinamento.py
│
├── pseudocodigos/
│   └── pseudocodigos.md
│
├── README.md
├── resumo_sessao.md
├── indice_estudos.md
├── glossario.md
└── materiais_complementares.md
```

---

## Referências

- [micrograd — Andrej Karpathy](https://github.com/karpathy/micrograd)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Poli-USP — Álgebra Linear](https://www.youtube.com/playlist?list=PLO3hBdfBc4pFef1zn1oZyYXLomL9MiX-C)
- NeoSpace — aula_ldm.pdf (material interno)