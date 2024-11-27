# **ECG Analysis Notebooks**

This directory contains Jupyter notebooks demonstrating the implementation, training, and evaluation of various models for ECG signal analysis. Each notebook serves a specific purpose, from building models to evaluating their performance.

---

## **Available Notebooks**

### **1. ResNet Notebooks**
- **`resnet.ipynb`**:
  - Implements the standard ResNet architecture.
  - Demonstrates training and validation processes.
- **`resnet_a.ipynb`**:
  - Implements ResNet with an attention mechanism.
  - Highlights how attention improves feature weighting.
- **`resnet-encoder.ipynb`**:
  - Implements ResNet with a focus on feature extraction for encoder tasks.
  - Designed for downstream tasks like transfer learning.

---

### **2. CNN-BiLSTM Notebooks**
- **`cnn_bilstm.ipynb`**:
  - Demonstrates the basic CNN-BiLSTM architecture.
  - Explains feature extraction using convolutional layers and temporal modeling with BiLSTM layers.
- **`cnn_bilstm_a.ipynb`**:
  - Extends CNN-BiLSTM with attention blocks.
  - Highlights the benefits of attention for temporal ECG feature analysis.

---

### **3. BiLSTM with Attention Notebook**
- **`bilstm_a.ipynb`**:
  - Builds a standalone BiLSTM architecture with attention for ECG classification.
  - Focuses on temporal feature modeling and leveraging attention for better interpretability.

---

### **4. Evaluation Notebook**
- **`evaluation.ipynb`**:
  - Provides a unified framework for evaluating all models on metrics such as accuracy, F1-score, precision, recall, and AUC.
  - Includes visualizations like confusion matrices and performance curves.

---

## **Usage Instructions**

1. **Setup**:
   - Ensure you have the required Python packages installed. Use the `requirements.txt` file from the repository for easy setup:
     ```bash
     pip install -r requirements.txt
     ```

2. **Run the Notebooks**:
   - Open any notebook in Jupyter Notebook, Jupyter Lab, or VS Code.
   - Example:
     ```bash
     jupyter notebook resnet.ipynb
     ```

3. **Data Requirements**:
   - Ensure the raw and processed ECG data (`physionet2017.csv` and `dwt_features_ecg.csv`) are available in the specified directory.
   - Update file paths in the notebooks if necessary.

4. **Reproducibility**:
   - Each notebook is designed to be standalone, allowing you to reproduce results for specific architectures.

---

## **Notebook Highlights**

| Notebook             | Focus                                | Key Features                          |
|-----------------------|--------------------------------------|---------------------------------------|
| `resnet.ipynb`        | ResNet Architecture                 | Residual learning, feature extraction |
| `resnet_a.ipynb`      | ResNet with Attention               | Feature weighting with attention      |
| `resnet-encoder.ipynb`| ResNet for Encoding Tasks           | Feature extraction for downstream ML  |
| `cnn_bilstm.ipynb`    | CNN-BiLSTM                          | Spatial and temporal modeling         |
| `cnn_bilstm_a.ipynb`  | CNN-BiLSTM with Attention           | Improved temporal feature focus       |
| `bilstm_a.ipynb`      | BiLSTM with Attention               | Temporal analysis and interpretability|
| `evaluation.ipynb`    | Model Evaluation                    | Unified performance analysis          |

---

## **Contributing**
Feel free to modify and improve the notebooks. If you’d like to contribute:
1. Fork the repository.
2. Create a branch for your changes.
3. Submit a pull request with detailed notes.

---

## **License**
This repository is licensed under the MIT License. Refer to the `LICENSE` file for details.

---

## **Contact**
For questions or feedback, create an issue or reach out to the repository maintainer.
