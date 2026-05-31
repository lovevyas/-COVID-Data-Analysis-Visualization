from src.numpy_ops   import CovidArrayLoader, NumpyAnalyzer
from src.pandas_Data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.EDAAnalyzer import EDAAnalyzer
 
def main():
    
    loader   = CovidArrayLoader()
    analyser = NumpyAnalyzer(loader)
    analyser.run() 
      
    data = DataLoader()
    data.summary() 
    
    cleaner = DataCleaner(data)
    clean_df=cleaner.clean()
    cleaner.summary()
    
    EDAanalyzer = EDAAnalyzer(clean_df)
    EDAanalyzer.EDA_Analysis()
    

 
if __name__ == "__main__":
    main()
 