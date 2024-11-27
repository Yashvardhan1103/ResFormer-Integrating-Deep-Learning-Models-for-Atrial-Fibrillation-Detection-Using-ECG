# **ResFormer: Integrating Deep Learning Models for Atrial Fibrillation Detection Using ECG**

## **Abstract**
Atrial fibrillation (AF), a serious and prevalent heart rhythm disorder, requires timely and accurate detection to prevent complications such as stroke. This study explores advanced deep-learning models for automated AF detection, comparing CNN-BiLSTM and ResNet architectures. ResNet's superior performance is further enhanced by integrating a Transformer Encoder mechanism, leveraging ResNet for spatial feature extraction and the Transformer for capturing long-term dependencies in ECG signals. This hybrid approach significantly improves accuracy and reliability in detecting AF.

---

## **Key Highlights**
1. **Problem Addressed**:
   - Atrial fibrillation detection from ECG signals.
   - Overcoming the limitations of manual analysis and traditional machine learning techniques.
2. **Proposed Solution**:
   - Hybrid architecture combining ResNet for spatial features and Transformer for temporal dependencies.
3. **Dataset**:
   - PhysioNet 2017 ECG dataset, featuring thousands of single-lead recordings labeled for various rhythm classes.

---

## **Methodology**
1. **Data Preprocessing**:
   - Standardization of ECG signals using methods such as wavelet denoising, R-R interval calculation, and PQRS complex extraction.
   - Features combined into structured datasets for training.
2. **Model Architectures**:
   - **CNN-BiLSTM**:
     - Combines convolutional layers for spatial features and BiLSTMs for temporal dependencies.
   - **ResNet**:
     - Employs residual blocks to handle deep architectures and mitigate vanishing gradients.
   - **ResNet + Transformer**:
     - Integrates ResNet's spatial feature extraction with Transformer's multi-head attention for long-term temporal dependency modeling.
3. **Training Setup**:
   - Models trained with binary cross-entropy loss and the Adam optimizer.
   - Class weights used to address dataset imbalances.
   - Callbacks like early stopping and learning rate reduction applied.

---

## **Results**
- **ResNet + Transformer** demonstrated the best performance:
  - **Accuracy**: 93.25%
  - **Precision**: 96% (Normal), 70% (AF)
  - **Recall**: 95% (Normal), 75% (AF)
  - **F1-Score**: 96% (Normal), 73% (AF)
- Comparative Performance:
  | Model                 | Accuracy | Sensitivity | Specificity |
  |-----------------------|----------|-------------|-------------|
  | **CNN-BiLSTM**        | 87.58%   | 89.33%      | 85.75%      |
  | **ResNet**            | 92.00%   | 90.61%      | 93.31%      |
  | **ResNet + Transformer** | 93.25% | 92.30%      | 94.11%      |

---

## **Conclusion**
This research demonstrates the effectiveness of hybrid deep learning models for AF detection:
- ResNet + Transformer outperforms standalone models by leveraging both spatial and temporal dependencies.
- Future work will focus on addressing data imbalances and enhancing sensitivity to improve clinical applicability.

---

## **Reference**
If using this research, please cite:
> ResFormer: Integrating Deep Learning Models for Atrial Fibrillation Detection Using ECG

Refer to the full research paper for in-depth methodologies, experimental setups, and findings.
