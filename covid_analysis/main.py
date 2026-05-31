from src.numpy_ops   import CovidArrayLoader, NumpyAnalyzer
from src.pandas_Data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.EDAAnalyzer import EDAAnalyzer
from src.visualization import Visualizer
 
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
    viz = Visualizer(clean_df)
    viz.start_visuaization()

 
if __name__ == "__main__":
    main()
 