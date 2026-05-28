from src.numpy_ops   import CovidArrayLoader, NumPyAnalyzer
 
 
def main():
    # Task 1 - NumPy
    loader   = CovidArrayLoader()
    analyser = NumPyAnalyzer(loader)
    analyser.run()   
 
 
if __name__ == "__main__":
    main()
 