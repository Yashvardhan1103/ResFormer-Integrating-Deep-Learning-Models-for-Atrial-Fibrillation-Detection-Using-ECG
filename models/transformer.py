import tensorflow as tf
from tensorflow.keras import layers

def transformer_encoder_dual(inputs, head_size, num_heads, ff_dim, dropout=0.3):
    """
    Defines a dual-attention Transformer encoder block designed for processing two inputs
    (e.g., R-R intervals and PQRS complexes) with separate attention mechanisms.

    Args:
        inputs: List of two tensors - [R-R intervals, PQRS complexes].
        head_size: Size of each attention head.
        num_heads: Number of attention heads.
        ff_dim: Dimension of the feedforward network.
        dropout: Dropout rate for regularization.

    Returns:
        Output tensor after applying dual attention and feedforward transformations.
    """
    # Multi-head self-attention for R-R intervals
    attention_rr = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs[0], inputs[0])
    attention_rr = layers.LayerNormalization(epsilon=1e-6)(attention_rr)  # Normalize attention output
    attention_rr = layers.Add()([attention_rr, inputs[0]])  # Add residual connection

    # Multi-head self-attention for PQRS complexes
    attention_pqrs = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs[1], inputs[1])
    attention_pqrs = layers.LayerNormalization(epsilon=1e-6)(attention_pqrs)  # Normalize attention output
    attention_pqrs = layers.Add()([attention_pqrs, inputs[1]])  # Add residual connection

    # Manually broadcast R-R intervals to match PQRS complex shape
    # (tile the R-R tensor along the time steps of PQRS complex input)
    expanded_rr = tf.tile(attention_rr, [1, inputs[1].shape[1], 1])

    # Combine the attention outputs from R-R intervals and PQRS complexes
    combined_attention = layers.Concatenate()([expanded_rr, attention_pqrs])

    # Feedforward network after the combined attention
    x_ff = layers.Dense(ff_dim, activation="relu")(combined_attention)  # First feedforward layer
    x_ff = layers.Dropout(dropout)(x_ff)  # Dropout for regularization

    # Project back to the original combined feature dimension
    x_ff = layers.Dense(combined_attention.shape[-1])(x_ff)
    x_ff = layers.LayerNormalization(epsilon=1e-6)(x_ff)  # Normalize output of feedforward layer

    # Add residual connection to combined attention output
    return layers.Add()([x_ff, combined_attention])

