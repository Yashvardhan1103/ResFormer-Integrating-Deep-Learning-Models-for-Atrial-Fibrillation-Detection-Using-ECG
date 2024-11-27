# **ECG Model Results**

This directory contains the evaluation results of various deep learning models for ECG classification. The results include classification reports, confusion matrices, and training performance metrics for each model.

---

## **Results Overview**

### **1. CNN-BiLSTM**
- **Classification Report**: `cnnbilstm_classification_report.png`
  - Shows precision, recall, F1-score, and support for each class.
  - Overall accuracy: **77%**.
- **Confusion Matrix**: `cnnbilstm_best_conf_matrix.png`
  - Provides a normalized confusion matrix visualizing true vs. predicted labels.

---

### **2. ResNet**
- **Classification Report**: `resnet_classification_report.png`
  - Highlights model performance across different classes.
  - Overall accuracy: **75%**.
- **Training Loss**: `resnet_epoch_loss.png`
  - Tracks the loss over epochs during training.
- **Training Accuracy**: `resnet_epoch_accuracy.png`
  - Tracks accuracy improvements over epochs.
- **Confusion Matrix**: `resnet_best_conf_matrix.png`
  - Displays a normalized confusion matrix for the best-performing ResNet model.

---

### **3. ResNet + Encoder**
- **ROC Curve**: `AUC-ROC_resnet+encoder.png`
  - Demonstrates the model's ability to distinguish between classes (AUC = **0.94**).
- **Confusion Matrix**: `resnet+encoder_conf_matrix.png`
  - Highlights true vs. predicted labels for ResNet + Encoder.

---

## **File Details**

### **1. Classification Reports**
- Contain detailed evaluation metrics:
  - **Precision**: Proportion of correctly predicted positive observations.
  - **Recall**: Proportion of actual positives correctly predicted.
  - **F1-Score**: Harmonic mean of precision and recall.
  - **Support**: Number of true instances for each class.

### **2. Confusion Matrices**
- Provide a visual summary of true vs. predicted labels.
- Normalized percentages make it easier to interpret model performance across imbalanced datasets.

### **3. Training Metrics**
- Loss and accuracy plots help visualize the training progress:
  - **Epoch Loss**: Indicates convergence and overfitting trends.
  - **Epoch Accuracy**: Shows how model accuracy improves during training.

### **4. ROC Curve**
- Measures the model's ability to classify instances across thresholds.
- AUC (Area Under the Curve) reflects the overall model performance (closer to 1 is better).

---

## **Performance Highlights**

| Model               | Accuracy | Key Observations                                   |
|---------------------|----------|---------------------------------------------------|
| **CNN-BiLSTM**      | 77%      | Strong performance on majority classes.          |
| **ResNet**          | 75%      | Gradual improvement in training; overfitting risk.|
| **ResNet + Encoder**| 94% (AUC)| Best overall classification ability (AUC = 0.94).|

---

## **Usage Instructions**
- View individual result files for more details:
  - Classification reports for text-based metrics.
  - Confusion matrices for visual performance summaries.
  - Training plots for insights into model optimization.
  - ROC curves for threshold-based evaluation.

---

## **Contributing**
Feel free to add new results or update existing ones. If you’d like to contribute:
1. Fork the repository.
2. Add your changes.
3. Submit a pull request with a summary of updates.

---

## **License**
This repository is licensed under the MIT License. Refer to the `LICENSE` file for details.

---

## **Contact**
For questions or feedback, create an issue or reach out to the repository maintainer.
