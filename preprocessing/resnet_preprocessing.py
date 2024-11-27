import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
def load_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    ecg_signals = data.iloc[:, :-2].values.reshape(-1, 2000, 1)
    labels = data.iloc[:, -1].values

    # Normalize the ECG signals
    scaler = StandardScaler()
    ecg_signals = scaler.fit_transform(ecg_signals.reshape(-1, 2000)).reshape(-1, 2000, 1)
    return ecg_signals, labels

# Split the dataset into training and testing sets
def split_data(ecg_signals, labels):
    return train_test_split(ecg_signals, labels, test_size=0.2, stratify=labels, random_state=42)
