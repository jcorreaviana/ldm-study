# Glossário — Machine Learning do Zero ao Transformer
> Termos organizados por área, do mais fundamental ao mais avançado.

---

## Matemática

**Derivada**
Mede a inclinação de uma função num ponto específico — o quanto a saída muda quando a entrada muda um pouquinho. Em ML, usada para calcular como cada parâmetro afeta o erro do modelo.

**Derivada parcial (∂L/∂w)**
Derivada de uma função com múltiplas variáveis em relação a uma delas, mantendo as outras fixas. Em ML: ∂L/∂w diz o quanto a perda muda se o peso w mudar.

**Regra da cadeia**
Regra para derivar funções compostas: dL/dx = (dL/du) · (du/dx). É o mecanismo matemático por trás do backpropagation.

**Gradiente**
Vetor de derivadas parciais — um valor por parâmetro. Aponta a direção em que a função de perda cresce mais rápido. O gradiente descendente anda na direção oposta.

**Produto escalar (dot product)**
Operação entre dois vetores: W · X = Σ wᵢxᵢ. Multiplica par a par e soma. É a operação fundamental de um neurônio: z = W · X + b.

**Multiplicação de matrizes**
C[i][j] = soma(A[i][k] × B[k][j]). Linha de A vezes coluna de B. Requer que o número de colunas de A seja igual ao número de linhas de B.

**Transposta**
Operação que troca linhas por colunas de uma matriz. Kᵀ[i][j] = K[j][i]. Usada no mecanismo de atenção: Q · Kᵀ.

**Número de Euler (e)**
Constante matemática ≈ 2.71828. Base da exponencial natural. Aparece na sigmoide e no softmax. math.exp(x) calcula eˣ.

---

## Normalização e Dados

**Z-score**
Normalização que transforma um valor em número de desvios padrão acima/abaixo da média: z = (x - μ) / σ. Garante que features de escalas diferentes fiquem comparáveis.

**Desvio padrão (σ)**
Mede o quanto os valores se afastam da média: σ = √(média dos quadrados das diferenças). Usado no z-score para escalar os dados.

**One-hot encoding**
Representa uma categoria como vetor binário com apenas um "1": 'asia' → [0, 0, 0, 1, 0]. Usado quando categorias são mutuamente exclusivas.

**Multi-hot encoding**
Variação do one-hot onde múltiplos "1" são permitidos. Usado quando um exemplo pode pertencer a várias categorias ao mesmo tempo.

**Feature engineering**
Processo manual de criar variáveis (features) a partir de dados brutos. Principal limitação do pipeline clássico de ML — rígido a mudanças de mercado.

**Event stream**
Sequência de eventos de um cliente ordenada por tempo. Base do LDM — ao invés de features agregadas, o modelo recebe a sequência bruta de eventos.

---

## Funções de Perda

**MSE (Mean Squared Error)**
L = (1/n) · Σ(y_prev - y_real)². Função de perda para regressão. Penaliza erros proporcionalmente. Quanto menor, melhor.

**Cross-Entropy Loss**
Função de perda para classificação. Penaliza erros de confiança de forma exponencial via logaritmo. Quanto mais confiante o modelo e mais errado, maior a penalidade.

**Binary Cross-Entropy (BCE)**
Variação da cross-entropy para classificação binária (0 ou 1): L = -[y × log(score) + (1-y) × log(1-score)]. Usada para fraude, churn, spam.

**Categorical Cross-Entropy**
Variação para classificação com N classes. Usada em LLMs para prever o próximo token entre 50.000+ palavras: L = -Σ yᵢ × log(pᵢ).

**Convergência**
Momento em que a perda para de cair significativamente entre épocas — o modelo encontrou um mínimo. Identificada visualmente por um platô na curva de perda.

**Platô**
Região da curva de perda onde ela para de diminuir. Indica que o modelo convergiu ou está preso num mínimo local.

---

## Treinamento

**Forward pass**
Passagem dos dados pela rede da entrada até a saída, calculando a previsão e a perda. Constrói o grafo computacional usado no backward.

**Backward pass (backpropagation)**
Percorre o grafo computacional de trás para frente aplicando a regra da cadeia. Calcula os gradientes de cada parâmetro em relação à perda.

**Gradiente descendente**
Algoritmo que usa os gradientes para ajustar os parâmetros na direção que diminui a perda: w ← w - η · ∂L/∂w. Repete até convergir ao mínimo.

**Learning rate (η)**
Tamanho do passo do gradiente descendente. Muito alto: instável, pode divergir. Muito baixo: converge devagar. Valores típicos: 0.001 a 0.1.

**Época (epoch)**
Uma passagem completa pelo dataset de treinamento. Ao final de cada época os pesos são atualizados.

**Batch**
Subconjunto dos dados processado de uma vez antes de atualizar os pesos. Batch maior = mais rápido, mais memória.

**Early stopping**
Para o treinamento quando o MSE de validação para de melhorar por `patience` épocas consecutivas. Evita overfitting.

**Patience**
Número de épocas sem melhora no conjunto de validação antes de o early stopping parar o treinamento.

**Overfitting**
Modelo decorou o dataset de treino. Perfeito no treino, ruim no teste. Detectado por: MSE treino << MSE teste, ou AUROC treino >> AUROC teste.

**Underfitting**
Modelo não aprendeu nem o básico. Ruim tanto no treino quanto no teste. Modelo muito simples ou dados insuficientes.

**Vanishing gradient**
Gradiente que desaparece ao ser propagado por muitas camadas. Parâmetros das primeiras camadas nunca são atualizados. Resolvido por residual connections.

**Exploding gradient**
Gradiente que cresce exponencialmente ao passar por muitas camadas. Causa overflow numérico. Resolvido por gradient clipping e He initialization.

**Gradient clipping**
Limita o valor máximo do gradiente durante o treinamento: grad = max(-limite, min(limite, grad)). Evita exploding gradient.

**He initialization**
Inicialização de pesos calibrada pela profundidade da rede: escala = √(2/n_entradas). Recomendada para ReLU e Leaky ReLU.

**Symmetry breaking**
Inicializar pesos com valores aleatórios diferentes entre neurônios. Sem isso, todos os neurônios aprendem a mesma coisa.

---

## Técnicas de Learning Rate

**Warmup**
Estratégia de aumentar o learning rate gradualmente no início do treinamento. Evita instabilidade quando os parâmetros ainda são aleatórios.

**Cooldown**
Estratégia de diminuir o learning rate gradualmente no final do treinamento. Permite afinar os parâmetros com precisão perto do mínimo.

**Cosine Annealing**
Scheduler que diminui o learning rate seguindo uma curva de cosseno: começa alto, cai suavemente, chega próximo a zero. Padrão em transformers e LLMs.

**ReduceLROnPlateau**
Reduz o learning rate automaticamente quando a perda para de melhorar por N épocas. Equivalente ao Early Stopping, mas para o lr.

**Scheduler de learning rate**
Estratégia para variar o lr ao longo do treinamento. Evita passos grandes perto do mínimo e reduz custo computacional.

**AdamW**
Otimizador adaptativo que ajusta o learning rate individualmente por parâmetro. Combina momentum (m) e variância (v) dos gradientes com weight decay correto. Padrão em LLMs modernos.

---

## Funções de Ativação

**ReLU (Rectified Linear Unit)**
f(x) = max(0, x). Retorna x se positivo, 0 se negativo. Introduce não-linearidade. Derivada: 1 se z>0, 0 se z≤0.

**Leaky ReLU**
f(x) = x se x>0, α·x se x≤0. Variação do ReLU onde α (ex: 0.01) evita que o gradiente seja completamente zero. Resolve o dying ReLU.

**Dying ReLU**
Neurônio que fica permanentemente inativo (z≤0 para todos os exemplos). Gradiente zero → pesos nunca atualizados. Leaky ReLU como solução.

**Sigmoid (sigmoide)**
f(z) = 1/(1+e⁻ᶻ). Transforma qualquer número em valor entre 0 e 1. Usada na saída para problemas de classificação binária (ex: fraude ou não).

**Softmax**
Transforma um vetor de scores em probabilidades que somam 1: eˣⁱ / Σeˣʲ. Usada na saída para classificação múltipla e no mecanismo de atenção.

---

## Avaliação

**R² (coeficiente de determinação)**
Mede quanto da variação real o modelo explica: R² = 1 - SS_res/SS_tot. R²=1 perfeito, R²=0 equivale a prever sempre a média.

**Matriz de confusão**
Tabela com os 4 resultados possíveis de um classificador binário: TP (acertou positivo), FP (falso alarme), FN (fraude passou), TN (acertou negativo).

**Precision**
TP / (TP + FP). Dos que o modelo acusou como positivos, quantos eram de verdade? "Quando bloquei, acertei?"

**Recall**
TP / (TP + FN). De todos os positivos reais, quantos o modelo encontrou? "Encontrei todas as fraudes?"

**F1 Score**
Média harmônica entre Precision e Recall: F1 = 2×(P×R)/(P+R). Única métrica para avaliar equilíbrio entre os dois.

**AUROC**
Area Under the ROC Curve. Mede a qualidade geral do modelo independente do limiar. 0.5=aleatório, 1.0=perfeito. Bom para datasets balanceados.

**AUPRC**
Area Under the Precision-Recall Curve. Melhor para datasets desbalanceados (fraude tem 0.17% de casos positivos).

**Limiar (threshold)**
Valor de corte aplicado ao score para tomar a decisão. Score ≥ limiar → positivo. Ajustável conforme o custo de cada erro.

---

## Arquitetura de Redes Neurais

**Neurônio**
Unidade básica: z = W·X + b, h = ativação(z). Recebe entradas, aplica pesos e viés, passa por função de ativação.

**Camada oculta**
Camada de neurônios entre a entrada e a saída. Cada camada aprende uma representação diferente dos dados.

**Parâmetros treináveis**
Pesos (W) e vieses (b) ajustados pelo gradiente descendente durante o treinamento. Não confundir com hiperparâmetros (learning rate, n_camadas).

**Feed-forward layer**
Duas camadas densas com relu no meio aplicadas a cada token independentemente: d_model → 4×d_model → d_model. Processa e transforma a informação combinada pela atenção.

**Residual connection**
Conexão que soma a entrada de uma camada com sua saída: saída = transformação(entrada) + entrada. Resolve vanishing gradient. Base do transformer.

**Layer normalization**
Z-score aplicado nas ativações dentro de cada camada: x' = (x - μ) / σ. Mantém gradientes em escala controlada. Padrão nos transformers.

---

## Transformer e LDM

**Token**
Unidade mínima de entrada do transformer. Em NLP: palavra ou subpalavra. No LDM: evento do cliente (transação, login, reclamação).

**Embedding**
Vetor de N dimensões que representa um token. Tokens similares ficam próximos no espaço vetorial. Parâmetro treinável.

**Positional encoding**
Vetor somado ao embedding para indicar a posição do token na sequência. Sem isso o transformer trata a sequência como desordenada.

**Self-attention**
Mecanismo que calcula, para cada token, o quanto cada outro token da sequência é relevante. Permite capturar dependências de longo alcance.

**Q, K, V (Query, Key, Value)**
Q = o que o token procura. K = o que cada token oferece. V = conteúdo real do token. scores = Q·Kᵀ/√d → softmax → ×V.

**Multi-head attention**
N cabeças de atenção em paralelo, cada uma com seus próprios Wq, Wk, Wv. Cada cabeça aprende um aspecto diferente — a especialização emerge do treinamento.

**Bloco transformer**
Unidade repetida N vezes: [Multi-Head Attention + Residual + LayerNorm] + [Feed-Forward + Residual + LayerNorm]. Entrada e saída sempre com mesma dimensão d_model.

**LDM (Large Data Model)**
Transformer treinado diretamente no event stream bruto — sem feature engineering manual. Unifica múltiplos modelos especializados num único modelo.

**Mean tyranny**
Limitação do pipeline clássico: médias apagam picos comportamentais críticos. 3 reclamações em 2 dias ≠ 3 reclamações em 60 dias — a média é igual.

**Knowledge imprinting**
Bias humano embutido nas features do pipeline clássico. O modelo confirma hipóteses existentes ao invés de descobrir padrões novos.

**TabNet**
Transformer para dados tabulares estáticos. Usa atenção sobre features. Proxy do LDM para datasets sem sequência temporal.

---

## Fine-tuning e Adaptação de Modelos

**Pré-treino (pre-training)**
Treinamento inicial do modelo em grande volume de dados sem labels. Aprende representações gerais. Feito uma vez, caro — tratado como ativo estratégico.

**Fine-tuning**
Ajuste fino de um modelo pré-treinado para uma tarefa específica. Mais rápido e preciso que treinar do zero.

**LoRA (Low-Rank Adaptation)**
Técnica de fine-tuning eficiente que congela os pesos originais e adiciona matrizes pequenas treináveis: W_novo = W_original + A × B. Reduz em até 99% os parâmetros treináveis.

**Rank (LoRA)**
Dimensão interna das matrizes A e B do LoRA. Rank baixo (1-4) = menos parâmetros, mais rápido. Rank alto (64+) = mais expressivo, mais caro. Sweet spot típico: 4-16.

**QLoRA (Quantized LoRA)**
Evolução do LoRA que comprime o modelo base de float16 para int4 (4x menos memória), mantendo A e B em float16. Permite fine-tuning de modelos grandes em GPUs consumer.

**Modelo base (frozen weights)**
Pesos do modelo pré-treinado congelados durante o fine-tuning. No QLoRA comprimidos para int4. Representam o conhecimento geral do pré-treino.

**Adaptação (A × B)**
Matrizes treináveis do LoRA que representam a diferença entre o modelo geral e o especializado. Ficam em float16 para manter precisão no treinamento.

**Data drift catastrófico**
Mudança estrutural no comportamento dos dados que invalida o modelo base. Ex: chegada do Pix mudando os padrões de transação. Pode justificar retreinar do zero.

**Retreino do zero**
Necessário quando: pré-treino ruim, mudança estrutural do domínio, drift catastrófico, viés sistemático ou arquitetura inadequada. Custo muito maior que fine-tuning.

**RLHF (Reinforcement Learning from Human Feedback)**
Técnica de fine-tuning que usa feedback humano para alinhar o modelo com preferências e valores. Usada no GPT-4, Claude e outros LLMs modernos.

---

## Variantes do Transformer

**BERT (Bidirectional Encoder Representations from Transformers)**
Transformer encoder-only treinado para preencher tokens mascarados: "o gato [MASK] o peixe" → "comeu". Lê a sequência nos dois sentidos simultaneamente. Ideal para compreensão de texto: classificação, extração de entidades, similaridade semântica.

**GPT (Generative Pre-trained Transformer)**
Transformer decoder-only treinado para prever o próximo token da esquerda para a direita. Ideal para geração de texto: completar, escrever, conversar. Base do ChatGPT e Claude.

**Masked Language Model (MLM)**
Objetivo de treinamento do BERT. Mascara tokens aleatórios da sequência e treina o modelo para prevê-los usando o contexto dos dois lados. Força o modelo a entender o contexto bidirecional.

**Encoder-only**
Arquitetura transformer que processa a sequência inteira de uma vez nos dois sentidos. Ex: BERT. Bom para tarefas de compreensão.

**Decoder-only**
Arquitetura transformer que processa tokens da esquerda para a direita, um por vez. Ex: GPT, Claude, Llama. Bom para geração de texto.

**Encoder-Decoder**
Arquitetura transformer com dois componentes: encoder processa a entrada, decoder gera a saída. Ex: T5, BART. Bom para tradução e resumo.