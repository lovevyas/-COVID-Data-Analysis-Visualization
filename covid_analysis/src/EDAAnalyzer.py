class EDAAnalyzer:
    def __init__(self, df):
        self.df = df

    def display(self, data):
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"\n{key}: {value}")
        else:
            print(data)

    def status_summary(self):
        df = self.df

        print("\nTotal stats")
        total_data = {
            "Total Confirmed": df["Confirmed"].sum(),
            "Total Deaths": df["Deaths"].sum(),
            "Total Recovered": df["Recovered"].sum()
        }
        self.display(total_data)

        print("\nWHO Region analysis")
        region_group = (
            df.groupby("WHO Region")
              .agg({
                  "Confirmed": "sum",
                  "Deaths": "sum",
                  "Recovered": "sum",
              })
              .sort_values(by="Confirmed", ascending=False)
        )
        self.display(region_group)

    def top_10_countries(self):
        return (
            self.df
            .sort_values(by="Confirmed", ascending=False)
            .head(10)
        )

    def highest_active(self):
        return (
            self.df
            .sort_values(by="Active", ascending=False)
            .head(5)
        )

    def EDA_Analysis(self):
        print("\nEDA ")
        
        self.status_summary()

        print("\nTop 10 Countries:")
        print(self.top_10_countries()[["Country/Region", "Confirmed"]].reset_index(drop=True))

        print("\nTop 5 Active Cases:")
        print(self.highest_active()[["Country/Region", "Active"]])

        