# Guia de Diagnóstico de Treinamento
> Referência rápida para interpretar loss, gradiente e métricas durante o treino.
> Construído a partir dos projetos práticos de fraude.

---

## Os Três Critérios

Durante o treinamento você monitora sempre três coisas em conjunto:

```
1. loss (perda)      →  o quanto o modelo está errando
2. gradiente         →  o quanto os pesos precisam ser ajustados
3. métrica de negócio →  AUPRC, recall, F1 — o que realmente importa
```

**Regra geral:**
```
o que importa não é o valor absoluto de nenhum dos três
mas a RELAÇÃO entre eles e a TENDÊNCIA ao longo das épocas
```

---

## O que cada um representa

**Loss (perda):**
```
mede o erro do modelo na função de perda escolhida (BCE, FocalLoss, MSE)
alta no início → normal (pesos aleatórios)
deve cair consistentemente ao longo das épocas
loss baixa desde o início → suspeito (pode ser mínimo local)
```

**Gradiente:**
```
mede o quanto os pesos precisam ser ajustados
alto no início → modelo aprendendo ativamente
vai diminuindo conforme a loss cai
≈ 0 no final → modelo convergiu (ou travou)
```

**Métrica de negócio:**
```
AUPRC, recall, F1 — o que realmente importa para o problema
deve subir junto com a queda da loss
se loss cai mas métrica não sobe → problema nas features ou arquitetura
```

---

## Os 4 Cenários

### Cenário 1 — Convergência Real ✅

```
época   loss    gradiente   AUPRC    interpretação
  1     3.912   0.8420      0.320    início — pesos aleatórios
  2     2.341   0.6103      0.510    aprendendo rápido
  3     1.187   0.3891      0.680    gradiente caindo
  5     0.521   0.1234      0.820    acelerando
  8     0.183   0.0412      0.940    desacelerando
 12     0.071   0.0089      0.985    refinando
 15     0.038   0.0021      0.997    quase convergiu ← parar aqui
```

**Diagnóstico:**
```
loss:      alta → baixa    ✓  caindo consistentemente
gradiente: alto → ≈ 0      ✓  acompanha a loss
métrica:   baixa → alta    ✓  subindo junto

decisão:   PARAR — convergência real
           early stopping correto
```

**Exemplo real:** transformer card testing (loss 0.221 → 0.008, AUPRC 1.000)

---

### Cenário 2 — Mínimo Local / Platô ❌

```
época   loss    gradiente   AUPRC    interpretação
  1     0.001   0.0003      0.0016   loss já baixa — suspeito!
  2     0.001   0.0002      0.0016   gradiente sumindo rápido
  3     0.001   0.0001      0.0016   platô
  5     0.001   0.0001      0.0016   gradiente ≈ 0 mas AUPRC ruim
```

**Diagnóstico:**
```
loss:      baixa desde o início  ⚠️  suspeito — não caiu, já começou baixa
gradiente: pequeno desde início  ⚠️  nunca teve gradiente significativo
métrica:   praticamente zero     ✗   modelo não discrimina nada

decisão:   NÃO é convergência — é mínimo local disfarçado
           modelo encontrou "solução preguiçosa":
           dar score igual para tudo
```

**Por que a loss ficou baixa?**
```
dataset com 0.12% de fraudes
modelo que chuta tudo como legítimo:
  erra só 0.12% → loss muito baixa
  mas AUPRC = 0.0016 → não detectou nenhuma fraude
```

**Exemplo real:** transformer PaySim sem features do remetente

**O que investigar:**
```
→  features têm sinal discriminativo? (rodar baseline primeiro)
→  pos_weight está adequado?
→  learning rate muito baixo?
→  bug no pipeline de dados?
```

---

### Cenário 3 — Aprendendo Normalmente ✅

```
época   loss    gradiente   AUPRC    interpretação
  1     3.912   0.8420      0.320    início normal
  5     0.521   0.1234      0.820    aprendendo bem
 10     0.183   0.0412      0.940    continuando
```

**Diagnóstico:**
```
loss alta no início → NORMAL
o que importa: está caindo a cada época?
gradiente alto no início → NORMAL
o que importa: está diminuindo junto com a loss?

decisão:   CONTINUAR treinando
           loss alta no início não é problema
           o problema seria loss alta que NÃO CAI
```

---

### Cenário 4 — Modelo Travado desde o Início ❌

```
época   loss    gradiente   AUPRC    interpretação
  1     0.693   0.0012      0.510    loss estranha — quase log(2)
  5     0.689   0.0009      0.509    quase não mudou
 12     0.687   0.0007      0.510    caindo muito devagar
 15     0.687   0.0006      0.510    vai travar em loss alta
```

**Diagnóstico:**
```
loss:      0.693 → 0.687   ✗  quase não caiu em 15 épocas
gradiente: 0.0012 → 0.0006 ✗  pequeno desde o início
métrica:   0.510 → 0.510   ✗  não saiu do aleatório

decisão:   PARAR e diagnosticar
           modelo travado desde o início
```

**O que 0.693 significa:**
```
BCE de uma moeda justa = log(2) = 0.693
→  modelo prevendo 50% para tudo
→  não aprendeu absolutamente nada
```

**Causas mais comuns:**
```
learning rate muito baixo  →  passos minúsculos, nunca sai do lugar
features sem sinal         →  nada para aprender
bug no pipeline            →  labels embaralhados, features erradas
pos_weight muito alto      →  overflow → nan → loss não finita
```

---

## Quadro Resumo

```
         loss      gradiente   métrica   diagnóstico
ex 1:    alta→baixa  alto→≈0   baixa→alta  convergência real      ✅
ex 2:    baixa(fixo) ≈0        baixa       mínimo local           ❌
ex 3:    alta→↓      alto→↓    baixa→↑     aprendendo normalmente ✅
ex 4:    alta(fixo)  pequeno   baixa       travado                ❌
```

---

## Checklist de Diagnóstico

```
loss caiu bastante?
  sim →  continua
  não →  quanto tempo treinou? se > 10 épocas sem queda → problema

gradiente está caindo junto com a loss?
  sim →  normal
  não →  learning rate inadequado ou bug no backward

métrica de negócio está subindo?
  sim →  modelo está aprendendo o que importa
  não →  loss pode estar caindo mas por razão errada
         (ex: chutar tudo como legítimo em dataset desbalanceado)

loss baixa desde a época 1?
  suspeito →  verificar se modelo não está chutando tudo igual
              rodar distribuição de scores (fraude vs legítimo)
              diferença < 0.01 → mínimo local
```

---

## Relação com Learning Rate

```
o objetivo NÃO é:  learning rate → 0
o objetivo É:      gradiente → 0  (consequência da convergência)

learning rate:  tamanho do passo — permanece fixo ou decai com scheduler
gradiente:      direção e magnitude do ajuste — vai a zero quando converge

loss baixa + gradiente ≈ 0 + métrica boa  →  para com confiança
loss baixa + gradiente ≈ 0 + métrica ruim →  mínimo local — diagnosticar
loss alta  + gradiente alto               →  continua treinando
loss alta  + gradiente pequeno            →  travado — diagnosticar
```

---

## Exemplos Reais dos Projetos

| Projeto | Cenário | Loss | Gradiente | AUPRC | Causa |
|---|---|---|---|---|---|
| Card testing | Convergência real | 0.221→0.008 | alto→≈0 | 1.000 | features + label corretos |
| PaySim transformer | Mínimo local | 0.001 fixo | ≈0 desde início | 0.0016 | features do remetente ausentes |
| PaySim (nan) | Travado | nan | nan | — | features sem z-score (overflow) |
| Credit card | Convergência real | caiu | caiu | 0.705 | transformer tabular funcionou |
EOF