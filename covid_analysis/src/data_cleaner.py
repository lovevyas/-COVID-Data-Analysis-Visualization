import numpy as np
import pandas as pd
from src.pandas_Data_loader import DataLoader
class DataCleaner:
    def __init__(self, df):
        self.loader   = df
        self.df       = self.loader.load().copy()        
        
    def clean(self):
        df = self.df
        df = df.drop_duplicates()
        df = df.replace([float("inf"),-float("inf")],np.nan)
        df = df.fillna(0)
        
        self.df = df
        return df        
    def summary(self):        
        info = {
            "Shape": self.df.shape,                                    
            "Duplicate rows": self.df.duplicated().sum(),
            "Null Values":  self.df.isnull().sum()
        }                
        print("\n---- Cleaner Data Summary ----")
        for key, value in info.items():
            print(f"\n{key}:")
            print(value)
        
if __name__ == "__main__":
    loader = DataLoader()
    cleaner = DataCleaner(loader)
    cleaner.summary()
    cleaner.clean()
    cleaner.summary()