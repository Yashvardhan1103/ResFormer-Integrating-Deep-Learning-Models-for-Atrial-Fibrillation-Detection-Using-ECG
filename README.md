# **ECG Signal Analysis and Classification**

This project provides a complete pipeline for ECG signal classification using deep learning. It includes raw and processed data, preprocessing scripts, model implementations, Jupyter notebooks, and evaluation results.

---

## **Project Structure**

### **1. Data**
- **Folder**: `data/`
- **Description**:
  - Contains raw and processed ECG data.
  - Raw data includes unprocessed ECG signals.
  - Processed data contains features extracted from ECG signals, such as wavelet coefficients, R-R intervals, and PQRS complexes.
- **Files**:
  - `physionet2017.csv`: Raw ECG signals with labels.
  - `dwt_features_ecg.csv`: Processed ECG features for machine learning.
- **Details**:
  - **Raw Data**:
    - Columns: Time-series signal values with a label column.
    - Example:
      | Signal1 | Signal2 | ... | Label |
      |---------|---------|-----|-------|
      | 0.1     | 0.2     | ... | 0     |
  - **Processed Data**:
    - Contains extracted features like PQRS complexes, R-R intervals, wavelet variance, and entropy.

---

### **2. Models**
- **Folder**: `models/`
- **Description**:
  - Contains Python scripts implementing various deep learning architectures for ECG classification.
- **Files**:
  - `resnet.py`: Implements the ResNet model.
  - `resnet_A.py`: ResNet with attention.
  - `cnn_bilstm.py`: Basic CNN-BiLSTM model.
  - `cnn_bilstm_a.py`: CNN-BiLSTM with attention.
  - `transformer.py`: Transformer-based model.
- **Model Highlights**:
  - **ResNet**:
    - Deep residual learning for feature extraction.
    - Includes attention-based variants.
  - **CNN-BiLSTM**:
    - Combines convolutional feature extraction with BiLSTM for sequence modeling.
    - Includes attention-enhanced variants.
  - **Transformer**:
    - Uses multi-head self-attention for long-range dependency modeling.

---

### **3. Notebooks**
- **Folder**: `notebooks/`
- **Description**:
  - Jupyter notebooks demonstrating model training, evaluation, and experimentation.
- **Files**:
  - `resnet.ipynb`: ResNet training and evaluation.
  - `resnet_a.ipynb`: ResNet with attention.
  - `resnet-encoder.ipynb`: ResNet for feature extraction tasks.
  - `cnn_bilstm.ipynb`: CNN-BiLSTM training and evaluation.
  - `cnn_bilstm_a.ipynb`: CNN-BiLSTM with attention.
  - `bilstm_a.ipynb`: BiLSTM with attention for temporal analysis.
  - `transformer_resnet16.ipynb`: Hybrid Transformer + ResNet model.
  - `evaluation.ipynb`: Unified model evaluation.

---

### **4. Preprocessing**
- **Folder**: `preprocessing/`
- **Description**:
  - Scripts for preprocessing ECG data, tailored to different models.
- **Files**:
  - `cnn_bi_lstm_preprocessing.py`: Prepares data for CNN-BiLSTM.
  - `resnet_preprocessing.py`: Prepares data for ResNet.
- **Functionality**:
  - Load and normalize ECG signals.
  - Extract features like PQRS complexes and R-R intervals.
  - Split data into training and testing sets.

---

### **5. Results**
- **Folder**: `results/`
- **Description**:
  - Contains visualizations and performance metrics for all models.
- **Files**:
  - **Classification Reports**:
    - `cnnbilstm_classification_report.png`: CNN-BiLSTM classification report.
    - `resnet_classification_report.png`: ResNet classification report.
  - **Confusion Matrices**:
    - `cnnbilstm_best_conf_matrix.png`: Best CNN-BiLSTM confusion matrix.
    - `resnet_best_conf_matrix.png`: Best ResNet confusion matrix.
    - `resnet+encoder_conf_matrix.png`: Confusion matrix for ResNet + Encoder.
  - **Training Metrics**:
    - `resnet_epoch_loss.png`: Training loss for ResNet.
    - `resnet_epoch_accuracy.png`: Training accuracy for ResNet.
  - **ROC Curve**:
    - `AUC-ROC_resnet+encoder.png`: ROC curve for ResNet + Encoder.
