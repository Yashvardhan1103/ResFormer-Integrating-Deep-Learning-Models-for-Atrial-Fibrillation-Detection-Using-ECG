import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, GRU, Bidirectional, Dense, Dropout, BatchNormalization, Multiply, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

def attention_block(inputs):
    """
    Self-attention mechanism for the LSTM/GRU output
    
    Args:
        inputs (tf.Tensor): Input tensor of shape (batch_size, time_steps, features).
    
    Returns:
        tf.Tensor: Output tensor after applying attention mechanism.
    """
    # Compute attention scores using a Dense layer
    attention_scores = Dense(1, activation='tanh')(inputs)  # Shape: (batch_size, time_steps, 1)

    # Apply softmax to get attention weights
    attention_weights = tf.nn.softmax(attention_scores, axis=1)  # Shape: (batch_size, time_steps, 1)

    # Multiply attention weights with the inputs
    output_attention = Multiply()([inputs, attention_weights])  # Broadcasting across features, Shape: (batch_size, time_steps, features)

    return output_attention

def create_bi_lstm_model(time_step, num_features, lstm_units=128, dropout_rate=0.3, num_classes=4):
    """
    Function to define a Bi-LSTM/GRU model with attention for ECG classification.
    
    Args:
        input_shape (tuple): Shape of the input data (time steps, features).
        lstm_units (int): Number of LSTM/GRU units for the Bidirectional layers.
        dropout_rate (float): Dropout rate to avoid overfitting.
        num_classes (int): Number of output classes.
    
    Returns:
        model (tf.keras.Model): Compiled Bi-LSTM/GRU model with attention.
    """
    # Input layer
    inputs = Input(shape=(time_step, num_features))
    
    # Bi-GRU layers with attention
    x = Bidirectional(GRU(units=256, return_sequences=True))(inputs)
    x = attention_block(x)
    x = Bidirectional(GRU(units=128, return_sequences=True))(x)
    x = attention_block(x)
    
    # Global pooling layer
    x = GlobalMaxPooling1D()(x)
    
    # Fully connected layers
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=64, activation='relu')(x)
    
    # Output layer (Softmax for multi-class classification)
    outputs = Dense(units=num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=SigmoidFocalCrossEntropy(),  # Focal Loss for imbalanced classes
                  metrics=['accuracy'])
    
    return model
