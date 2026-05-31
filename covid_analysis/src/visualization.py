import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


class Visualizer:
    def __init__(self, df):
        self.df = df
    
    def _finalize_plot(self, filename, title, xlabel=None, ylabel=None, rotate_xticks=False):
        plt.title(title)

        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

        if rotate_xticks:
            plt.xticks(rotation=90)

        plt.tight_layout()

        save_path = OUTPUT_DIR / f"{filename}.png"
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

        plt.show()


    def top_confirmed(self):
        df = self.df.sort_values(by="Confirmed", ascending=False).head(20)

        plt.figure()
        plt.bar(df["Country/Region"], df["Confirmed"])

        self._finalize_plot(
            filename="top_confirmed_cases",
            title="Top 20 Countries by Confirmed Cases",
            xlabel="Country",
            ylabel="Confirmed Cases",
            rotate_xticks=True
        )

   
    def bar_region_deaths(self):
        region = self.df.groupby("WHO Region")["Deaths"].sum().sort_values()

        plt.figure()
        region.plot(kind="bar")

        self._finalize_plot(
            filename="deaths_by_region",
            title="Deaths by WHO Region",
            xlabel="Region",
            ylabel="Deaths"
        )
   
    def histogram_active(self):
        plt.figure()
        self.df["Active"].plot(kind="hist", bins=20)

        self._finalize_plot(
            filename="active_distribution",
            title="Distribution of Active Cases",
            xlabel="Active Cases",
            ylabel="Frequency"
        )

    
    def pie_recovered(self):
        region = (
            self.df.groupby("WHO Region")["Recovered"]
            .sum()
            .sort_values(ascending=False)
        )

        plt.figure()
        region.plot(kind="pie", autopct="%1.1f%%")

        self._finalize_plot(
            filename="recovered_by_region",
            title="Recovered Cases by Region",
            ylabel=""
        )

    
    def scatter_confirmed_deaths(self):
        plt.figure()
        plt.scatter(self.df["Confirmed"], self.df["Deaths"])

        self._finalize_plot(
            filename="confirmed_vs_deaths",
            title="Confirmed vs Deaths",
            xlabel="Confirmed",
            ylabel="Deaths"
        )

    
    def heatmap_corr(self):
        corr = self.df.corr(numeric_only=True)

        plt.figure()
        sns.heatmap(corr, annot=True, cmap="coolwarm")

        self._finalize_plot(
            filename="correlation_heatmap",
            title="Correlation Heatmap"
        )

    
    def start_visuaization(self):
        print("\n--- Generating Visualizations ---")

        self.top_confirmed()
        self.bar_region_deaths()
        self.histogram_active()
        self.pie_recovered()
        self.scatter_confirmed_deaths()
        self.heatmap_corr()       