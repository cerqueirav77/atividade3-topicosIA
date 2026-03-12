# Transformer Decoder From Scratch — LAB 3

## Descrição

Implementação dos blocos matemáticos centrais do Decoder da arquitetura Transformer,
baseada no paper *"Attention Is All You Need"* (Vaswani et al., 2017).

O laboratório cobre três componentes essenciais para a geração de texto:
a **Máscara Causal**, o **Cross-Attention** e o **Loop de Inferência Auto-Regressivo**.

## Como Rodar

**Pré-requisitos:** Python 3.x e NumPy.

1. Crie e ative um ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale as dependências:
```bash
pip install numpy
```

3. Execute o decoder:
```bash
python3 decoder.py
```

4. Desative o ambiente virtual ao terminar (opcional):
```bash
deactivate
```

## Estrutura do Laboratório

### Tarefa 1 — Máscara Causal (Look-Ahead Mask)

Durante o treinamento, a frase de destino completa entra no Decoder de uma vez.
Para impedir que o token na posição `i` "olhe para o futuro" (posição `i+1`),
injetamos uma máscara antes do Softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

A matriz `M` contém `0` na diagonal e triângulo inferior, e `-∞` no triângulo superior.
Após o Softmax, as posições futuras se tornam estritamente `0.0`.

### Tarefa 2 — Cross-Attention (Ponte Encoder-Decoder)

Diferente do Self-Attention, o Cross-Attention cruza duas sequências:

- **Query (Q)** vem do estado atual do Decoder
- **Keys (K) e Values (V)** vêm da saída do Encoder

Isso permite que o Decoder consulte a "memória" da frase de origem a cada passo
da geração. Não há máscara causal aqui — o modelo pode olhar a frase do Encoder
por completo.

### Tarefa 3 — Loop de Inferência Auto-Regressivo

O modelo gera texto uma palavra por vez em um laço `while`:

1. Recebe a sequência gerada até agora
2. Passa pelo Decoder e projeta para o tamanho do vocabulário
3. Aplica `argmax` para selecionar o token mais provável
4. Adiciona o token à sequência
5. Para imediatamente ao gerar `<EOS>`

## Exemplo de Output
```
Tarefa 1 — Máscara Causal (seq_len=5):
[[  0. -inf -inf -inf -inf]
 [  0.   0. -inf -inf -inf]
 [  0.   0.   0. -inf -inf]
 [  0.   0.   0.   0. -inf]
 [  0.   0.   0.   0.   0.]]

Validação — triângulo superior é estritamente 0.0? True

Tarefa 2 — Cross-Attention:
Shape encoder_output : (1, 10, 512)
Shape decoder_state  : (1,  4, 512)
Shape saída cross    : (1,  4, 512)
Validação — shape esperado (1, 4, 512)? True

Tarefa 3 — Loop de inferência iniciado em ["<START>"]
Passo 01 — token gerado: 'palavra_XXXX'
...
Token <EOS> detectado! Geração encerrada.
```

## Referência

Vaswani, A. et al. **Attention Is All You Need**, 2017.
https://arxiv.org/abs/1706.03762

## Auxiliado por

Implementação desenvolvida com auxílio do Claude (Anthropic) como ferramenta
de suporte ao aprendizado, conforme permitido pelo contrato pedagógico da disciplina.