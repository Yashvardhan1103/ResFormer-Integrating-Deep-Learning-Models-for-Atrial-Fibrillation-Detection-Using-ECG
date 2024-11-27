import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    ecg_data = data.iloc[:, :-2].values  # All columns except the last two
    labels = data.iloc[:, -1].values     # Last column as labels

    # Normalize and reshape data for LSTM
    scaler = StandardScaler()
    ecg_data_scaled = scaler.fit_transform(ecg_data)
    ecg_data_scaled = ecg_data_scaled.reshape(-1, ecg_data.shape[1], 1)  # Reshape for LSTM

    # Encode labels as integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    return ecg_data_scaled, labels, len(le.classes_)

def split_data(ecg_data_scaled, labels):
    return train_test_split(
        ecg_data_scaled, labels, test_size=0.2, stratify=labels, random_state=42)
