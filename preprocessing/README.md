# **Preprocessing for ECG Data**

This directory contains preprocessing scripts designed to prepare ECG datasets for training various deep learning models, including CNN-BiLSTM and ResNet architectures. The scripts include functions for loading, normalizing, reshaping, and splitting data.

---

## **Available Scripts**

### **1. CNN-BiLSTM Preprocessing**
- **File**: `cnn_bi_lstm_preprocessing.py`
- **Description**:
  - Prepares data specifically for CNN-BiLSTM models, ensuring compatibility with sequential input requirements.
- **Key Functions**:
  - **`load_data(csv_file_path)`**:
    - Loads ECG data from a CSV file.
    - Standardizes features using `StandardScaler`.
    - Reshapes data to a 3D format (`[samples, time_steps, 1]`) suitable for LSTM layers.
    - Encodes class labels as integers.
  - **`split_data(ecg_data_scaled, labels)`**:
    - Splits the dataset into training and testing sets with an 80-20 ratio.
    - Maintains label distribution using stratified sampling.

---

### **2. ResNet Preprocessing**
- **File**: `resnet_preprocessing.py`
- **Description**:
  - Tailored for ResNet models, focusing on reshaping ECG signals to match the input expectations of 1D convolutional layers.
- **Key Functions**:
  - **`load_data(csv_file_path)`**:
    - Loads ECG data from a CSV file.
    - Normalizes features using `StandardScaler`.
    - Reshapes signals to a 3D format (`[samples, time_steps, 1]`) for Conv1D layers.
  - **`split_data(ecg_signals, labels)`**:
    - Splits the dataset into training and testing sets with an 80-20 ratio.
    - Maintains label distribution using stratified sampling.

---

## **Usage Instructions**

1. **Prepare Your Data**:
   - Ensure the raw ECG data is saved as a CSV file with the following structure:
     - Columns: Signal values and a label column at the end.
     - Example:
       | Signal1 | Signal2 | ... | Label |
       |---------|---------|-----|-------|
       | 0.1     | 0.2     | ... | 0     |
       | 0.3     | 0.4     | ... | 1     |

2. **Run the Preprocessing Script**:
   - Import the appropriate preprocessing script for your model:
     ```python
     from cnn_bi_lstm_preprocessing import load_data, split_data
     ecg_data, labels, num_classes = load_data("path/to/data.csv")
     X_train, X_test, y_train, y_test = split_data(ecg_data, labels)
     ```

3. **Outputs**:
   - `ecg_data`: Scaled and reshaped ECG signals.
   - `labels`: Encoded class labels.
   - `X_train`, `X_test`, `y_train`, `y_test`: Split data for training and testing.

---

## **File Comparison**

| Script                     | Target Model  | Key Features                                     |
|----------------------------|---------------|-------------------------------------------------|
| `cnn_bi_lstm_preprocessing.py` | CNN-BiLSTM   | StandardScaler, 3D reshaping for LSTM, label encoding |
| `resnet_preprocessing.py`      | ResNet        | StandardScaler, 3D reshaping for Conv1D layers     |

---

## **Contributing**
Feel free to enhance the preprocessing scripts. If you’d like to contribute:
1. Fork the repository.
2. Add your changes.
3. Submit a pull request with detailed notes.

---

## **License**
This repository is licensed under the MIT License. Refer to the `LICENSE` file for details.

---

## **Contact**
For questions or feedback, create an issue or reach out to the repository maintainer.
