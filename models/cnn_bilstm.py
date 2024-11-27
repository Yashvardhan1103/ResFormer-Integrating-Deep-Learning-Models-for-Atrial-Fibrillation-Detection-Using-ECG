import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dropout, MaxPooling1D, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Input #type: ignore
from tensorflow.keras.models import Model #type: ignore


def conv_block_type1(x, num_filters, kernel_size, dropout_rate=0.3):
    x = Conv1D(num_filters, kernel_size, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(num_filters, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D()(x)
    return x

def conv_block_type2(x, num_filters, kernel_size, dropout_rate=0.3):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(num_filters, kernel_size, activation='relu', padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(num_filters, kernel_size, activation='relu', padding="same")(x)
    x = MaxPooling1D()(x)
    return x

def build_model(time_step, num_sensors, num_classes):
    input_shape = (time_step, num_sensors)
    inpt = Input(input_shape)
    x = Conv1D(32, 16, activation='relu')(inpt)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = conv_block_type1(x, 32, 16)
    x = conv_block_type2(x, 64, 16)
    x = conv_block_type2(x, 128, 8)
    x = conv_block_type2(x, 256, 4)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inpt, outputs=x)
    return model