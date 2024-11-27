# **ECG Dataset Documentation**

This repository contains datasets for ECG signal analysis, including **Raw Data** and **Processed Data**, designed for tasks like arrhythmia detection using features derived from ECG waveforms.

---

## **1. Raw Data (`physionet2017.csv`)**

### **Description**
This dataset contains raw ECG signals from the PhysioNet Challenge 2017 database. Each row represents a single ECG recording sampled at a specific frequency, with labels indicating the rhythm type.

### **Features**
- **Signal Columns**: Time-series data representing raw ECG waveforms. Each column corresponds to a sampled time point in the ECG signal.
- **Labels**:
  - `0`: Normal sinus rhythm.
  - `1`: Atrial fibrillation (AF).
  - (Other rhythm types are excluded during preprocessing.)
- **Sampling Frequency**: Signals are sampled at 500 Hz.

### **Structure**
| Column Name        | Description                                    |
|--------------------|------------------------------------------------|
| Signal1, ..., SignalN | Raw ECG signal values (time-series).          |
| Label              | Target class (0 for Normal, 1 for AF).         |

### **Usage**
This dataset is used as input for preprocessing steps, including:
1. **Resampling** to ensure uniform sampling frequency (e.g., 300 Hz).
2. **Denoising** using wavelet transforms.
3. **Feature extraction**, such as R-R intervals and PQRS complexes.

---

## **2. Processed Data (`dwt_features_ecg.csv`)**

### **Description**
This dataset contains features extracted from raw ECG signals after preprocessing. It is designed for machine learning models for arrhythmia detection.

### **Features**
The dataset includes both handcrafted features and time-series data extracted from ECG signals:
- **Wavelet Features**:
  - Coefficients derived using Discrete Wavelet Transform (DWT) for denoising and compression.
- **PQRS Complexes**:
  - Extracted waveform segments centered around R-peaks.
- **R-R Intervals**:
  - Time intervals between consecutive R-peaks, capturing heart rate variability.
- **Handcrafted Features**:
  - **Wavelet Variance**: Represents signal complexity.
  - **Entropy**: Measures randomness or irregularity in the signal.

### **Structure**
| Column Name        | Description                                  |
|--------------------|----------------------------------------------|
| PQRS1, ..., PQRSN  | Flattened PQRS complex data (e.g., 30 × 200).|
| RR1, ..., RR20     | Standardized R-R intervals.                 |
| Wavelet_Variance   | Variance of wavelet coefficients.           |
| Entropy            | Signal entropy.                             |
| Label              | Target class (0 for Normal, 1 for AF).      |

### **Preprocessing Steps**
1. **Resampling**: All signals were resampled to 300 Hz.
2. **Denoising**: DWT applied to remove noise.
3. **Feature Extraction**:
   - PQRS complexes extracted using R-peak detection.
   - R-R intervals calculated and standardized.
4. **Handcrafted Features**:
   - Wavelet variance and entropy computed for each signal.

---

## **General Notes**
1. Both datasets follow standard ECG analysis practices and are suitable for machine learning tasks.
2. **Ethical Use**:
   - These datasets should be used for educational and research purposes only.
   - Ensure compliance with data privacy and ethical guidelines when handling ECG data.

### **Citation**
If using this dataset, please cite the PhysioNet Challenge 2017 as the source of the raw data.

---

## **File Overview**
- `physionet2017.csv`: Raw ECG data with time-series waveforms and labels.
- `dwt_features_ecg.csv`: Processed ECG data with extracted features for ML models.
