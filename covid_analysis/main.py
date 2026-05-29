from src.numpy_ops   import CovidArrayLoader, NumPyAnalyzer
from src.pandas_Data_loader import DataLoader
 
def main():
    
    loader   = CovidArrayLoader()
    analyser = NumPyAnalyzer(loader)
    analyser.run()   
    data = DataLoader()
    data.summary() 
 
 
if __name__ == "__main__":
    main()
 