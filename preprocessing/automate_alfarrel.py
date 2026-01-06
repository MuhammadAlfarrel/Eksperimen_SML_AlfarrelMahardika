import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import kagglehub
import os

def preprocess_data():
    print("=== Mulai Preprocessing Dataset Besar (70k Data) ===")
    
    # 1. Load Data via KaggleHub
    print("Downloading dataset...")
    path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
    # Cari path file csv yang benar
    csv_file = os.path.join(path, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
# Tes trigger GitHub Actions
    df = pd.read_csv(csv_file)
    print(f"Dataset loaded. Original size: {df.shape}")
    
    # 2. Hapus Duplikat
    df = df.drop_duplicates()
    
    # 3. Handle Missing Values
    df.fillna(df.median(), inplace=True)
    
    # 4. Handle Outlier (BMI)
    Q1 = df['BMI'].quantile(0.25)
    Q3 = df['BMI'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['BMI'] < (Q1 - 1.5 * IQR)) | (df['BMI'] > (Q3 + 1.5 * IQR)))]
    
    # 5. Binning (BMI)
    bins = [0, 18.5, 24.9, 29.9, 100]
    labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels)
    
    # 6. Encoding
    df = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)
    
    # 7. Scaling
    # Kolom target: 'Diabetes_binary'
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_clean = pd.DataFrame(X_scaled, columns=X.columns)
    df_clean['Diabetes_binary'] = y.values
    
    # Save Output
    output_path = "diabetes_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"=== Selesai! Data bersih disimpan di: {output_path} ===")

if __name__ == "__main__":
    preprocess_data()
    
    # 2. Tes GitHub Actions (Automation Advance)