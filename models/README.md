# **ECG Models Documentation**

This folder contains the implementation of machine learning and deep learning models designed for ECG signal analysis. These models are built to classify arrhythmias and analyze ECG features such as PQRST complexes, R-R intervals, and handcrafted signal characteristics.

---

## **Available Models**

### **1. ResNet Model**
- **File**: `resnet.py`
- **Description**:
  - A Residual Neural Network (ResNet) adapted for 1D time-series data like ECG signals.
  - Utilizes skip connections to prevent vanishing gradients and enable the training of deep networks.
- **Key Features**:
  - Designed with multiple `Conv1D` layers and residual connections.
  - Includes L2 regularization and dropout to mitigate overfitting.
  - Employs `GlobalAveragePooling1D` to reduce the feature map to a fixed-size vector.
- **Use Case**:
  - Ideal for extracting features and classifying ECG signals.

---

### **2. CNN-BiLSTM Model**
- **File**: `cnn_bilstm.py`
- **Description**:
  - A hybrid model combining Convolutional Neural Networks (CNNs) for feature extraction and Bidirectional Long Short-Term Memory (BiLSTM) layers for capturing temporal dependencies.
- **Key Features**:
  - Stacked `Conv1D` layers with Batch Normalization, MaxPooling, and ReLU activation.
  - BiLSTM layer processes temporal relationships in ECG signals.
  - Dense layers for classification.
- **Use Case**:
  - Suitable for tasks requiring both spatial and temporal pattern recognition in ECG data.

---

### **3. Transformer Model**
- **File**: `transformer.py`
- **Description**:
  - A Transformer-based model leveraging self-attention mechanisms for sequence modeling.
  - Processes multiple inputs, such as R-R intervals and PQRS complexes, with separate attention heads.
- **Key Features**:
  - Multi-Head Attention for both waveform and temporal features.
  - Layer normalization and residual connections for stable learning.
  - Dense layers for post-attention feature extraction.
- **Use Case**:
  - Suitable for advanced ECG analysis where capturing long-range dependencies is critical.

---

## **Model Comparison**

| Model            | Architecture        | Strengths                                     | Use Cases                           |
|-------------------|---------------------|-----------------------------------------------|-------------------------------------|
| **ResNet**       | Deep residual CNN   | Prevents vanishing gradients, efficient feature extraction | Large-scale ECG classification      |
| **CNN-BiLSTM**   | CNN + BiLSTM        | Combines spatial and temporal features       | Sequence-based tasks, time-series data |
| **Transformer**  | Self-attention      | Captures long-range dependencies, dual input support | Advanced ECG analysis, multi-feature integration |

---

## **Model Files**

Each model has its dedicated Python script. The files are structured as follows:
- **`resnet.py`**:
  Contains the implementation of the ResNet model with customizable parameters like number of layers, filter size, and dropout rates.
- **`cnn_bilstm.py`**:
  Implements the CNN-BiLSTM hybrid model with modular convolution blocks and a BiLSTM layer for sequence processing.
- **`transformer.py`**:
  Defines the Transformer model with multi-head attention layers and a feedforward network.

---

## **Usage Instructions**

### **1. Import the Model**
Each model can be imported and instantiated in your scripts. Example:
```python
from resnet import build_resnet
from cnn_bilstm import build_model as build_cnn_bilstm
from transformer import transformer_encoder_dual

# Example usage
resnet_model = build_resnet(input_shape=(3000, 1), num_classes=2)
cnn_bilstm_model = build_cnn_bilstm(time_step=3000, num_sensors=1, num_classes=2)
transformer_model = transformer_encoder_dual(inputs=[rr_input, pqrs_input], head_size=64, num_heads=8, ff_dim=128)
