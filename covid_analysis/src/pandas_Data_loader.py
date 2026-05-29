import pandas as pd 
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "country_wise_latest_coviddata.csv"

class DataLoader:
    
    def __init__(self, path: Path = DATA_PATH):
        if not path.exists():
            raise FileNotFoundError(f"CSV not found at {path}")
        self.path = path
        self._df = None
    
    def load(self):
        if self._df is None:
            df = pd.read_csv(self.path)
            df.columns = df.columns.str.strip()
            self._df = df
        return self._df
    
    def summary(self):
        df = self.load()
        info = {
            "Shape": df.shape,
            "First 5 rows": df.head(),
            "Last 5 rows": df.tail(),
            "Data Types": df.dtypes,
            "Duplicate rows": df.duplicated().sum(),
            "Null Values":  df.isnull().sum()
        }                
        print("\n---- Data Summary ----")
        for key, value in info.items():
            print(f"\n{key}:")
            print(value)


if __name__ == "__main__":
    loader = DataLoader()
    loader.summary()