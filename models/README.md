# **ECG Models Documentation**

This folder contains the implementation of various deep learning models designed for ECG signal analysis, including classification and feature extraction. The models leverage state-of-the-art architectures such as ResNet, CNN-BiLSTM, and Transformer.

---

## **Available Models**

### **1. ResNet Model**
- **Files**:
  - `resnet.py`: Standard ResNet model.
  - `resnet_A.py`: ResNet model with an attention mechanism.
- **Description**:
  - **Standard ResNet**:
    - Implements deep residual blocks with `Conv1D`, skip connections, and pooling layers.
    - L2 regularization and dropout for preventing overfitting.
    - Ends with global average pooling and a dense layer for classification.
  - **ResNet with Attention**:
    - Adds an attention mechanism to emphasize significant features in the feature maps.
- **Use Case**:
  - Suitable for large-scale ECG classification tasks.

---

### **2. CNN-BiLSTM Model**
- **Files**:
  - `cnn_bilstm.py`: Basic CNN-BiLSTM model.
  - `cnn_bilstm_a.py`: CNN-BiLSTM model with an attention mechanism.
- **Description**:
  - **Basic CNN-BiLSTM**:
    - Combines convolutional layers for feature extraction with Bidirectional LSTM layers for sequence modeling.
    - Includes modular convolution blocks (`conv_block_type1`, `conv_block_type2`).
  - **CNN-BiLSTM with Attention**:
    - Introduces attention blocks after BiLSTM layers to highlight critical temporal features.
    - Uses Focal Loss to handle imbalanced datasets.
- **Use Case**:
  - Effective for tasks requiring spatial and temporal feature extraction in ECG data.

---

### **3. Transformer Model**
- **File**: `transformer.py`
- **Description**:
  - A Transformer-based model with dual self-attention mechanisms for processing multiple inputs (e.g., R-R intervals and PQRS complexes).
  - Uses multi-head attention layers and residual connections for stable learning.
- **Use Case**:
  - Best suited for advanced ECG analysis tasks, such as integrating multiple features and capturing long-range dependencies.

---

## **Model Comparison**

| Model                  | Architecture             | Strengths                                       | Use Cases                                |
|------------------------|--------------------------|------------------------------------------------|------------------------------------------|
| **ResNet**             | Deep residual CNN        | Efficient feature extraction, prevents vanishing gradients | Large-scale ECG classification          |
| **ResNet with Attention** | Residual CNN + Attention | Highlights important features                  | Advanced ECG feature analysis            |
| **CNN-BiLSTM**         | CNN + BiLSTM             | Combines spatial and temporal features         | Sequence modeling and time-series tasks |
| **CNN-BiLSTM with Attention** | CNN + BiLSTM + Attention | Adds focus to critical temporal features       | ECG signal classification with imbalance |
| **Transformer**        | Self-Attention           | Captures long-range dependencies               | Multi-feature ECG analysis              |

---

## **Model Files**

### **1. ResNet**
- **`resnet.py`**:
  - Implements the standard ResNet architecture with 16 residual blocks.
  - Fully configurable filters, kernel sizes, and pooling layers.
- **`resnet_A.py`**:
  - Enhances ResNet with an attention mechanism to weigh significant features.

### **2. CNN-BiLSTM**
- **`cnn_bilstm.py`**:
  - A hybrid architecture combining CNN layers with BiLSTM for sequential processing.
- **`cnn_bilstm_a.py`**:
  - Adds attention layers to the CNN-BiLSTM model for better feature focus.

### **3. Transformer**
- **`transformer.py`**:
  - A Transformer encoder with dual attention mechanisms for multi-input processing.

---

## **Usage Instructions**

### **1. Import the Models**
Each model can be imported from its respective file. Example:
```python
from resnet import build_resnet
from cnn_bilstm import build_model as build_cnn_bilstm
from transformer import transformer_encoder_dual

# Example usage
resnet_model = build_resnet(input_shape=(3000, 1), num_classes=2)
cnn_bilstm_model = build_cnn_bilstm(time_step=3000, num_sensors=1, num_classes=2)
transformer_model = transformer_encoder_dual(inputs=[rr_input, pqrs_input], head_size=64, num_heads=8, ff_dim=128)
