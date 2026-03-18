import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataIngestion:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}

    def load_data(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        return df

    def handle_missing_values(self, df):
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else '', inplace=True)
        return df

    def normalize_data(self, df, numeric_cols):
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df

    def encode_categorical(self, df, cat_cols):
        for col in cat_cols:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col].astype(str))
        return df

    def feature_engineering(self, df):
        if 'Amount' in df.columns:
            df['Amount_Log'] = np.log(df['Amount'] + 1)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
        return df