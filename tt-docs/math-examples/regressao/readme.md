# Regressão Linear com Gradiente Descendente

Exemplo didático de uma rede neural minúscula (uma única "unidade" linear)
aprendendo a função `peso = a * altura + b` — implementado de duas formas:

- `regressao_visual.py` — usando [micrograd](https://github.com/karpathy/micrograd) (autodiferenciação automática)
- `regressao_do_zero.py` — implementação manual, sem bibliotecas de ML

---

## Como rodar

```bash
# versão com micrograd e gráfico
pip install micrograd matplotlib
python3 regressao_visual.py

# versão do zero, sem dependências
python3 regressao_do_zero.py
```

`regressao_visual.py` gera o arquivo `regressao_altura_peso.png` com dois gráficos:
- a reta aprendida sobre os dados reais
- a curva de erro (perda) caindo ao longo das épocas, em escala log

---

## Arquivos

- `regressao_visual.py` — script com micrograd, gera gráfico PNG
- `regressao_do_zero.py` — script sem bibliotecas, derivadas feitas à mão
- `regressao_altura_peso.png` — saída gerada pelo script visual
- `paisagem_perda.png` — visualização da função de perda L(a,b) e o caminho do gradiente descendente

---

## Conceitos demonstrados

- Domínio (X) e imagem (Y) de uma função
- Parâmetros treináveis (`a` = peso, `b` = viés)
- Normalização de dados e por que ela é necessária
- Forward pass — calcular a previsão
- Função de perda MSE
- Derivadas parciais e regra da cadeia
- Gradiente descendente — atualização dos parâmetros
- Overfitting e generalização
- Early Stopping e divisão treino/validação/teste

---

## 1. O Problema

Dado um conjunto de exemplos conhecidos (dataset), queremos encontrar uma função
que consiga prever novos valores nunca vistos antes.

```
f: X → Y
domínio (X) = alturas      →      imagem (Y) = pesos
```

---

## 2. O Modelo

A função mais simples possível com parâmetros ajustáveis — a equação da reta:

```
f(x) = a * x + b

a  →  peso (weight)   →  controla a INCLINAÇÃO da reta
b  →  viés (bias)     →  controla a ALTURA base da reta (peso previsto na média)
x  →  entrada         →  altura normalizada
f(x) →  saída         →  peso previsto (inferência)
```

**Importante:** `a` e `b` não são os dados — são os parâmetros que o modelo
aprende. Os dados são `x` (entrada) e `y` (saída real).

---

## 3. Normalização

```
x' = x - média(x)
```

Por quê? Porque sem normalizar, o gradiente de `a` fica muito menor que o de `b`,
causando desequilíbrio — `b` converge rápido e `a` anda devagar.

Após normalizar, `x'` fica centrado em zero. A soma dos valores normalizados
é sempre zero — isso confirma que a normalização está correta.

---

## 4. Função de Perda — MSE

Mede a distância entre o que o modelo prevê e o que é real:

```
L(a,b) = (1/n) * Σ (f(xᵢ) - yᵢ)²
```

- Resultado sempre positivo (por causa do quadrado)
- Penaliza erros grandes mais do que erros pequenos
- Diferenciável em todo ponto (essencial para calcular o gradiente)
- **Objetivo: minimizar L** — quanto mais próximo de zero, melhor

```
MSE alto  →  modelo muito errado  →  longe do mínimo
MSE baixo →  modelo bem ajustado  →  próximo do mínimo
```

---

## 5. Derivada — a essência

A derivada mede a **inclinação** de uma função num ponto específico.

```
f(x) = x²   →   f'(x) = 2x
```

| x | f'(x) | significado |
|---|---|---|
| 1 | 2 | inclinação suave |
| 2 | 4 | inclinação moderada |
| 3 | 6 | inclinação íngreme |

**Regra:** quanto maior a derivada, mais longe do mínimo você está.

### Tipos de pontos numa curva

```
máximo    →  topo da montanha  ⋀  →  derivada = 0
mínimo    →  fundo do vale     ⋁  →  derivada = 0  ← o que queremos
inflexão  →  mudança de forma     →  derivada ≠ 0
```

---

## 6. Derivadas Parciais — dL/da e dL/db

Como a perda depende de dois parâmetros (`a` e `b`), calculamos uma derivada
para cada — mantendo o outro fixo:

```
∂L/∂a  →  "quanto L muda se eu variar a?"
∂L/∂b  →  "quanto L muda se eu variar b?"
```

Aplicando a regra da cadeia (eᵢ = f(xᵢ) - yᵢ):

```
∂L/∂a = (1/n) * Σ 2 * eᵢ * xᵢ
∂L/∂b = (1/n) * Σ 2 * eᵢ
```

---

## 7. Gradiente Descendente

Algoritmo que usa a derivada para ajustar os parâmetros na direção do mínimo:

```
a = a - lr * ∂L/∂a
b = b - lr * ∂L/∂b
```

**O sinal negativo é fundamental** — garante que você sempre anda morro abaixo:

```
gradiente positivo  →  mínimo à esquerda  →  a diminui ←
gradiente negativo  →  mínimo à direita   →  a aumenta →
```

**learning rate (lr)** controla o tamanho do passo:

```
lr muito alto  →  passa o mínimo, oscila
lr muito baixo →  converge devagar
lr ideal       →  desce suave até o mínimo
```

---

## 8. O Ciclo Completo de Treinamento

```
para cada época:
    1. forward pass    →  calcula y_prev = f(x)
    2. função de perda →  calcula MSE = (1/n) * Σ(y_prev - y_real)²
    3. gradientes      →  calcula ∂L/∂a e ∂L/∂b (regra da cadeia)
    4. atualização     →  a = a - lr * ∂L/∂a
                          b = b - lr * ∂L/∂b
```

---

## 9. Overfitting e Generalização

```
MSE treino baixo + MSE teste alto  →  overfitting (decorou, não aprendeu)
MSE treino baixo + MSE teste baixo →  modelo saudável (generalizou)
```

**Early Stopping:** para o treinamento quando o MSE de validação para de
melhorar por `patience` épocas consecutivas.

**Divisão do dataset:**

```
treino     (~70%)  →  o modelo aprende aqui
validação  (~15%)  →  ajuste de hiperparâmetros e Early Stopping
teste      (~15%)  →  usado UMA só vez, para medir a qualidade final
```

---

## 10. O Código Comentado

```python
# DATASET — pares (domínio → imagem) conhecidos
alturas = [1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90]  # X (domínio)
pesos   = [55.0, 59.0, 64.0, 70.0, 75.0, 81.0, 85.0]  # Y (imagem)
n = len(alturas)  # número de exemplos

# NORMALIZAÇÃO — x' = x - média(x)
media_x = sum(alturas) / n
xs = [x - media_x for x in alturas]  # entradas normalizadas
ys = pesos                            # saídas reais

# MODELO — f(x) = a*x + b
def modelo(x, a, b):
    return a * x + b

# FUNÇÃO DE PERDA — MSE: L(a,b) = (1/n) * Σ (f(xᵢ) - yᵢ)²
def perda(a, b):
    total = 0.0
    for x, y in zip(xs, ys):       # zip combina xs e ys em pares (xᵢ, yᵢ)
        y_prev = modelo(x, a, b)    # previsão do modelo
        total += (y_prev - y) ** 2  # erro quadrático — sempre positivo
    return total / n                # média dos erros = MSE

# GRADIENTES — ∂L/∂a e ∂L/∂b (regra da cadeia aplicada manualmente)
def gradientes(a, b):
    dL_da = 0.0
    dL_db = 0.0
    for x, y in zip(xs, ys):
        e = modelo(x, a, b) - y  # erro do ponto i: previsão - real
        dL_da += 2 * e * x       # contribuição para ∂L/∂a
        dL_db += 2 * e           # contribuição para ∂L/∂b
    return dL_da / n, dL_db / n

# TREINAMENTO — gradiente descendente
a = 1.0    # peso inicial (chute arbitrário)
b = 1.0    # viés inicial (chute arbitrário)
lr = 0.1   # learning rate — tamanho do passo
epocas = 1000

for epoca in range(epocas):
    L = perda(a, b)                   # 1. calcula a perda
    dL_da, dL_db = gradientes(a, b)   # 2. calcula os gradientes
    a = a - lr * dL_da                # 3. atualiza peso
    b = b - lr * dL_db                # 4. atualiza viés
```

---

## 11. Convergência de a e b

```
b converge rápido (~época 60)
    → ∂L/∂b = 2 * e  (sem multiplicar por x)
    → gradiente grande → passo grande → chega logo na média de y

a converge devagar (até época ~1000)
    → ∂L/∂a = 2 * e * x  (x pequeno após normalização)
    → gradiente pequeno → passo pequeno → afina a inclinação devagar
```

No gráfico da paisagem de perda, isso aparece como um caminho em "L":
- trecho vertical → b descendo rapidamente
- trecho horizontal → a ajustando devagar

---

## 12. Correlação Matemática ↔ Código

| Matemática | Código | Significado |
|---|---|---|
| Domínio X | `alturas` | entradas conhecidas |
| Imagem Y | `pesos` | saídas conhecidas |
| x' = x - μ | `xs = [x - media_x ...]` | normalização |
| f(x) = a·x + b | `modelo(x, a, b)` | função do modelo |
| eᵢ = f(xᵢ) - yᵢ | `e = modelo(x,a,b) - y` | erro do ponto i |
| L = (1/n)·Σeᵢ² | `perda(a, b)` | função de perda MSE |
| ∂L/∂a | `dL_da` | derivada parcial de a |
| ∂L/∂b | `dL_db` | derivada parcial de b |
| a ← a - η·∂L/∂a | `a = a - lr * dL_da` | gradiente descendente |
| η (eta) | `lr` | learning rate |
| época | iteração do `for` | um ciclo completo de ajuste |