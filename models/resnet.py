from tensorflow.keras import layers, models, regularizers #type: ignore


def resnet_block(input_data, filters, kernel_size, pool_size, stride=1, dropout_rate=0.3, l2_lambda=0.001):
    """
    Defines a single residual block with L2 regularization, skip connections, BatchNorm, ReLU, and optional pooling.
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

    # Apply MaxPooling1D if pool_size > 1
    if pool_size > 1:
        x = layers.MaxPooling1D(pool_size=pool_size)(x)

    # Apply dropout
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)

    return x

def build_resnet(input_shape, num_classes, dropout_rate=0.3, l2_lambda=0.001):
    inputs = layers.Input(shape=input_shape)

    # ResNet blocks
    x = resnet_block(inputs, filters=32, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)  # Block 1
    x = resnet_block(x, filters=32, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 2
    x = resnet_block(x, filters=32, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 3 (Sequence Length: 16)
    x = resnet_block(x, filters=64, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 4
    x = resnet_block(x, filters=64, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 5 (Sequence Length: 7)
    x = resnet_block(x, filters=64, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 6
    x = resnet_block(x, filters=64, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)       # Block 7 (Sequence Length: 3)
    x = resnet_block(x, filters=128, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 8
    x = resnet_block(x, filters=128, kernel_size=16, pool_size=2, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 9 (Sequence Length: 1)

    # From here onwards, set pool_size=1 to prevent sequence length from becoming zero
    x = resnet_block(x, filters=128, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 10
    x = resnet_block(x, filters=128, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 11
    x = resnet_block(x, filters=256, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 12
    x = resnet_block(x, filters=256, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 13
    x = resnet_block(x, filters=256, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 14
    x = resnet_block(x, filters=256, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 15
    x = resnet_block(x, filters=256, kernel_size=16, pool_size=1, dropout_rate=dropout_rate, l2_lambda=l2_lambda)      # Block 16

   # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)

    if num_classes is not None:
        # Output layer for classification
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    else:
        # Return features without classification layer
        outputs = x

    model = models.Model(inputs, outputs)
    return model

