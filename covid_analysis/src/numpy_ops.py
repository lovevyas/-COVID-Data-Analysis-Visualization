"""
numpy_ops.py — Task 1: NumPy Operations
Pure NumPy only. Pandas enters in Task 2.
 
Classes
-------
CovidArrayLoader
NumPyAnalyzer
"""

import numpy as np
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "country_wise_latest_coviddata.csv"

# Debug helper — uncomment if getting FileNotFoundError
# print(f"Looking for CSV at: {DATA_PATH}")
# print(f"File exists: {DATA_PATH.exists()}")

class CovidArrayLoader:
    
    def __init__(self):
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"CSV not found at {DATA_PATH}\n"
            ) 
        self.matrix = np.genfromtxt(DATA_PATH, 
                                    delimiter=",", 
                                    skip_header=1, 
                                    usecols=(1,2,3,4), 
                                    dtype=np.float64, 
                                    filling_values=np.nan)
        self.countries = np.genfromtxt(DATA_PATH,
                                       delimiter=",",
                                       skip_header=1,
                                       usecols=(0,), # code repetition 
                                       dtype = "str")
        self.confirmed = self.matrix[:, 0]
        
    def __repr__(self):
        return f"CovidArrayLoader(shape={self.matrix.shape})"
    
class NumpyAnalyzer:
    def __init__(self, loader: CovidArrayLoader):
        self.matrix = loader.matrix
        self.countries = loader.countries
        self.confirmed = loader.confirmed
        self.cfr       = None
        self.normalised = None           
         
    #features    
    def show_1d_array(self):
        print("1. Single-Dimensional Array")
        a = self.confirmed
        print(f"Shape   : {a.shape}  |  Dtype: {a.dtype}  |  Ndim: {a.ndim}")
        print(f"First 5 : {a[:5]}")
        print(f"Total countries: {len(a)}")
        
    def show_2d_array(self):        
        print("2. Multi-Dimensional Array (ndarray)")
        m = self.matrix
        print(f"Shape   : {m.shape}  |  Ndim: {m.ndim}  |  Size: {m.size}")
        print(f"Columns : [Confirmed | Deaths | Recovered | Active]")
        print(f"First 3 rows:\n{m[:3]}")
        
    def show_math_ops(self):            
        print("\n3. Mathematical Operations")    
        c = self.confirmed
        stats = [("Sum", np.nansum), ("Mean", np.nanmean),
                 ("Median", np.nanmedian), ("Max", np.nanmax), 
                 ("Min", np.nanmin), ("Std", np.nanstd)]
        
        
        p25, p75 = np.nanpercentile(c, [25, 75])
        print("IQR:", p75 - p25)
        print("Cum sum:", np.nancumsum(c)[-1])
     
    def show_vectorized(self):
        print("4. Vectorized Operations\n")
        confirmed, death, recovered = (
            self.matrix[:,0], self.matrix[:,1], self.matrix[:,2]
        )
        self.cfr = np.where(confirmed > 0, (death/confirmed)*100, 0.0)
        rec_rate = np.where(confirmed > 0, (recovered/confirmed)*100, 0.0)
        
        print(f"CFR (Deaths/Confirmed × 100) — no loop, all 187 at once")
        print(f"Avg CFR      : {np.nanmean(self.cfr):.2f}%")
        print(f"Highest CFR  : {np.nanmax(self.cfr):.2f}%")
        print(f"Avg Recovery : {np.nanmean(rec_rate):.2f}%")
        
    def show_broadcasting(self):
        print("5. Broadcasting\n")
        m = self.matrix
        col_min, col_max = np.nanmin(m, axis=0), np.nanmax(m, axis=0)
        self.normalised = (m - col_min) / (col_max - col_min)
        print(f"matrix {m.shape}  ÷  col_max {col_max.shape}  → broadcasts automatically")
        print(f"Normalised range : {self.normalised.min():.2f} → {self.normalised.max():.2f}") # smip
        print(f"First row (normalised):\n{self.normalised[0].round(4)}")
    
    def show_slicing(self):
        print("6. Slicing & Indexing")    
        cases = self.confirmed
        countries = self.countries        
        
        print("First 5 cases:", cases[:5])
        print("Last country:", countries[-1])
        print("Last country cases:", cases[-1])        
        
        mask = cases > 10000
        print("Number of countries over 10k:", np.sum(mask))
                
        filtered_countries = countries[mask][:4]
        filtered_cases = cases[mask][:4]        
        for i in range(len(filtered_countries)):
            print(filtered_countries[i], "-", filtered_cases[i])
            
        print("Specific positions:")    
        for i in [0, 50, 100, 186]:
            print("Index", i, ":", countries[i], "-", cases[i])
         
        top_indexes = np.argsort(cases)[-5:]  
        top_indexes = top_indexes[::-1]      
        
        print("Top 5 Countries:")
        for i in range(5):
            idx = top_indexes[i]
            rank = i + 1
            print(rank, countries[idx], "-", cases[idx])
    
    def show_reshaping(self):
        print("7. Reshaping Arrays")
        matrix = self.matrix                
        print("Original shape:", matrix.shape)    
        
        flat_matrix = matrix.flatten()
        print("Flattened shape:", flat_matrix.shape)                
        
        reshaped_matrix = matrix.reshape(11, 17, 4)
        print("Reshaped (11, 17, 4) shape:", reshaped_matrix.shape)        
        
        transposed_matrix = matrix.T
        print("Transposed shape:", transposed_matrix.shape)    
 
    def show_splitting(self):
        print("8. Splitting Array")
        m = self.matrix
        
        def print_info(name, arr):
            print(f"{name}: {arr.shape}")            
        
        print("\nRow-wise split (array_split): ") 
               
        parts = np.array_split(m, 2)
        for part in parts:
            print_info("Part", part)
            
        print("\nVertical split (vsplit):")
        try:
            vparts = np.vsplit(m,9)
            for part in vparts:
                print_info(f"VPart", part)                    
        except:
            print(f"Current matrix shape is {m.shape}. It cannot be divided into 2 equal parts.")
            
        print("\nHorizontal split (hsplit):")
        cols = np.hsplit(m,4)
        names = ["Confirmed", "Deaths", "Recovered", "Active"]
        for name, col in zip(names, cols):
            print_info(name, col)
            
        print("\nCustom split at index: ")
        top, bottom = np.array_split(m, [170])
        print_info("Top", top)
        print_info("Bottom", bottom)
        
        
    # ── Run all ───────────────────────────────────────────────────────────────
 
    def run(self):
        
        print("  TASK 1 — NumPy Operations")        
 
        self.show_1d_array()
        self.show_2d_array()
        self.show_math_ops()
        self.show_vectorized()
        self.show_broadcasting()
        self.show_slicing()
        self.show_reshaping()
        self.show_splitting()
             
 
 
if __name__ == "__main__":
    loader   = CovidArrayLoader()
    analyser = NumpyAnalyzer(loader)
    analyser.run()
        
        
        
        
    