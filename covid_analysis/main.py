from src.numpy_ops   import CovidArrayLoader, NumpyAnalyzer
from src.pandas_Data_loader import DataLoader
from src.data_cleaner import DataCleaner
def main():
    
    loader   = CovidArrayLoader()
    analyser = NumpyAnalyzer(loader)
    analyser.run() 
      
    data = DataLoader()
    data.summary() 
    
    cleaner = DataCleaner(data)
    cleaner.clean()
    cleaner.summary()
    
 
 
if __name__ == "__main__":
    main()
 