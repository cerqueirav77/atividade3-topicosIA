import numpy as np

D_MODEL = 64
np.random.seed(42)


# Tarefa 1: Máscara Causal (Look-Ahead Mask)

def softmax(matriz: np.ndarray) -> np.ndarray:
    """Aplica softmax linha a linha com estabilidade numérica."""
    valores_deslocados = matriz - np.max(matriz, axis=-1, keepdims=True)
    exponenciais = np.exp(valores_deslocados)
    return exponenciais / np.sum(exponenciais, axis=-1, keepdims=True)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Cria a máscara causal (Look-Ahead Mask) para o Decoder.

    A parte triangular inferior e diagonal contém zeros (posições permitidas).
    A parte triangular superior contém -infinito (posições futuras bloqueadas).

    Args:
        seq_len (int): Tamanho da sequência.

    Returns:
        ndarray: Matriz quadrada (seq_len, seq_len) com a máscara causal.
    """
    mascara = np.full((seq_len, seq_len), -np.inf)
    mascara = np.tril(np.zeros((seq_len, seq_len))) + np.triu(mascara, k=1)
    return mascara


# Prova Real da Máscara
SEQ_LEN = 5

Q_ficticio = np.random.randn(SEQ_LEN, D_MODEL)
K_ficticio = np.random.randn(SEQ_LEN, D_MODEL)

mascara_causal = create_causal_mask(SEQ_LEN)

scores = Q_ficticio @ K_ficticio.T / np.sqrt(D_MODEL)
scores_mascarados = scores + mascara_causal
pesos_atencao = softmax(scores_mascarados)

print("=" * 55)
print("  Tarefa 1 — Máscara Causal (Look-Ahead Mask)")
print("=" * 55)
print(f"\nMáscara causal (seq_len={SEQ_LEN}):")
print(mascara_causal)
print("\nPesos de atenção após máscara (posições futuras = 0.0):")
print(np.round(pesos_atencao, 4))
print("\nValidação — triângulo superior é estritamente 0.0?", 
      np.all(np.triu(pesos_atencao, k=1) == 0.0))