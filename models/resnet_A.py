from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

# Residual block with attention
def resnet_block_with_attention(input_data, filters, kernel_size, pool_size, stride=1, dropout_rate=0.3, l2_lambda=0.001):
    """
    Defines a single residual block with L2 regularization, skip connections, BatchNorm, ReLU, optional pooling, and attention mechanism.
    """
    # First Conv1D layer with L2 regularization
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda))(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second Conv1D layer with L2 regularization
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)

    # Skip connection
    shortcut = layers.Conv1D(filters=filters, kernel_size=1, strides=stride, padding='same',
                             kernel_regularizer=regularizers.l2(l2_lambda))(input_data)
    shortcut = layers.BatchNormalization()(shortcut)

    # Add the skip connection
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    # Attention mechanism
    attention_scores = layers.Dense(1, activation='tanh')(x)
    attention_weights = tf.nn.softmax(attention_scores, axis=1)
    x = layers.Multiply()([x, attention_weights])

    # Apply MaxPooling1D if pool_size > 1
    if pool_size > 1:
        x = layers.MaxPooling1D(pool_size=pool_size)(x)

    # Apply dropout
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    return x

# Build the full ResNet model with attention
def build_resnet_with_attention(input_shape, num_classes, dropout_rate=0.3, l2_lambda=0.001):
    """
    Build the full ResNet model using residual blocks with attention mechanism, L2 regularization, and configurations based on the provided table.

    Parameters:
    - input_shape: tuple, shape of the input data (sequence_length, num_channels).
    - num_classes: int, number of output classes.
    - dropout_rate: float, dropout rate for Dropout layers.
    - l2_lambda: float, L2 regularization parameter.
    """
    inputs = layers.Input(shape=input_shape)

    # ResNet blocks with attention (16 blocks with L2 regularization)
    x = resnet_block_with_attention(inputs, filters=32, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)  # Block 1
    x = resnet_block_with_attention(x, filters=32, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 2
    x = resnet_block_with_attention(x, filters=32, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 3
    x = resnet_block_with_attention(x, filters=64, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 4
    x = resnet_block_with_attention(x, filters=64, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 5
    x = resnet_block_with_attention(x, filters=64, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 6
    x = resnet_block_with_attention(x, filters=64, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 7
    x = resnet_block_with_attention(x, filters=128, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 8
    x = resnet_block_with_attention(x, filters=128, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 9
    x = resnet_block_with_attention(x, filters=128, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 10
    x = resnet_block_with_attention(x, filters=128, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 11
    x = resnet_block_with_attention(x, filters=256, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 12
    x = resnet_block_with_attention(x, filters=256, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 13
    x = resnet_block_with_attention(x, filters=256, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 14
    x = resnet_block_with_attention(x, filters=256, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 15
    x = resnet_block_with_attention(x, filters=256, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 16

    # Global average pooling and dense layer for classification
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

# Example usage
input_shape = (2000, 1)  # Assuming input shape of 2000 time steps and 1 feature
num_classes = 4
model = build_resnet_with_attention(input_shape, num_classes)
model.summary()