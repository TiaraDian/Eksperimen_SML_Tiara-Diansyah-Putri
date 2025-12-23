import pandas as pd
from sklearn.preprocessing import StandardScaler
RAW_PATH = "Heart_Disease_Prediction_raw.csv"
PROCESSED_PATH = "preprocessing/Heart_Disease_Prediction_preprocessing.csv"

def preprocessing_data(df):
    """
    Fungsi untuk melakukan preprocessing data secara otomatis
    hingga data siap digunakan untuk pelatihan model.
    
    Tahapan:
    1. Handling outlier kolom Cholesterol (IQR method)
    2. Encoding target Heart Disease
    3. Standarisasi fitur numerik
    
    Parameters:
    df (pd.DataFrame): Data mentah
    
    Returns:
    pd.DataFrame: Data hasil preprocessing
    """

    # ===== 1. Handling Outlier Cholesterol =====
    Q1 = df['Cholesterol'].quantile(0.25)
    Q3 = df['Cholesterol'].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df['Cholesterol'] = df['Cholesterol'].clip(lower, upper)

    # ===== 2. Encoding Target =====
    df['Heart Disease'] = df['Heart Disease'].map({
        'Absence': 0,
        'Presence': 1
    })

    # ===== 3. Standarisasi Fitur Numerik =====
    num_cols = [
        'Age',
        'BP',
        'Cholesterol',
        'Max HR',
        'ST depression'
    ]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

if __name__ == "__main__":
    df_raw = pd.read_csv(RAW_PATH)
    df_processed = preprocessing_data(df_raw)

    # Simpan hasil di folder preprocessing
    df_processed.to_csv(PROCESSED_PATH, index=False)

    print("Dataset preprocessing berhasil dibuat")