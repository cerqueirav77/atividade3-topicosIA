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

# ── Tarefa 2: Cross-Attention (Ponte Encoder-Decoder) ─────────────────────────

D_MODEL_CROSS = 512
SEQ_LEN_FRANCES = 10
SEQ_LEN_INGLES = 4
BATCH_SIZE = 1

encoder_output = np.random.randn(BATCH_SIZE, SEQ_LEN_FRANCES, D_MODEL_CROSS)
decoder_state  = np.random.randn(BATCH_SIZE, SEQ_LEN_INGLES, D_MODEL_CROSS)

W_query_cross = np.random.randn(D_MODEL_CROSS, D_MODEL_CROSS)
W_key_cross   = np.random.randn(D_MODEL_CROSS, D_MODEL_CROSS)
W_value_cross = np.random.randn(D_MODEL_CROSS, D_MODEL_CROSS)


def cross_attention(encoder_out: np.ndarray, decoder_st: np.ndarray) -> np.ndarray:
    """
    Calcula o Cross-Attention entre o Encoder e o Decoder.

    O Decoder fornece as Queries e o Encoder fornece as Keys e Values,
    permitindo que o modelo consulte a memória da frase de origem.

    Args:
        encoder_out (ndarray): Saída do Encoder com shape (Batch, Seq_enc, D_MODEL).
        decoder_st  (ndarray): Estado do Decoder com shape (Batch, Seq_dec, D_MODEL).

    Returns:
        ndarray: Tensor de saída com shape (Batch, Seq_dec, D_MODEL).
    """
    Q = decoder_st  @ W_query_cross
    K = encoder_out @ W_key_cross
    V = encoder_out @ W_value_cross

    dimensao_k = K.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dimensao_k)

    # Sem máscara causal — o Decoder pode olhar a frase do Encoder por completo
    pesos_atencao = softmax(scores)
    saida = pesos_atencao @ V
    return saida


saida_cross = cross_attention(encoder_output, decoder_state)

print("\n" + "=" * 55)
print("  Tarefa 2 — Cross-Attention (Ponte Encoder-Decoder)")
print("=" * 55)
print(f"\nShape encoder_output : {encoder_output.shape}")
print(f"Shape decoder_state  : {decoder_state.shape}")
print(f"Shape saída cross    : {saida_cross.shape}")
print(f"\nValidação — shape esperado (1, {SEQ_LEN_INGLES}, {D_MODEL_CROSS})?",
      saida_cross.shape == (BATCH_SIZE, SEQ_LEN_INGLES, D_MODEL_CROSS))