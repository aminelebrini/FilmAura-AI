import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualisation:
    def __init__(self, data):
        self.data = data
        sns.set_theme(style="whitegrid")
        self.prepare_data()

    def prepare_data(self):
        self.data = self.data.copy()
        self.data["release_date"] = pd.to_datetime(self.data["release_date"], errors="coerce")
        self.data["release_year"] = self.data["release_date"].dt.year

    def extract_genres(self):
        genres = self.data["genres"].explode()
        return genres[genres.apply(lambda genre: isinstance(genre, dict))].apply(lambda genre: genre.get("name"))

    def plot_ratings_popularity(self):
        figure, axes = plt.subplots(1, 2)
        axes[0].hist(self.data["vote_average"])
        axes[0].set_title("Distribution des notes")
        axes[0].set_xlabel("Note moyenne")
        axes[0].set_ylabel("Nombre de films")
        axes[1].hist(self.data["popularity"])
        axes[1].set_title("Distribution de la popularite")
        axes[1].set_xlabel("Popularite")
        axes[1].set_ylabel("Nombre de films")
        plt.tight_layout()

        plt.show()

    def plot_genres(self):
        genres = self.extract_genres().dropna()
        genre_counts = genres.value_counts().sort_values()
        figure, axis = plt.subplots()
        axis.barh(genre_counts.index, genre_counts.values)
        axis.set_title("Nombre de films par genre")
        axis.set_xlabel("Nombre de films")
        axis.set_ylabel("Genre")
        plt.tight_layout()
        plt.show()
        return figure

    def plot_releases(self):
        releases = self.data.dropna(subset=["release_year"]).groupby("release_year").size()
        figure, axis = plt.subplots()
        axis.plot(releases.index, releases.values, marker="o")
        axis.set_title("Nombre de sorties par annee")
        axis.set_xlabel("Annee")
        axis.set_ylabel("Nombre de films")
        plt.tight_layout()
        plt.show()
        return figure

    def plot_runtime(self):
        figure, axis = plt.subplots()
        axis.hist(self.data["runtime"])
        axis.set_title("Distribution de la duree des films")
        axis.set_xlabel("Duree (minutes)")
        axis.set_ylabel("Nombre de films")
        plt.tight_layout()
        plt.show()
        return figure

    def plot_finance(self):
        finance_data = self.data[(self.data["budget"] > 0) | (self.data["revenue"] > 0)].copy()
        figure, axis = plt.subplots()
        axis.scatter(finance_data["budget"], finance_data["revenue"])
        axis.set_title("Relation entre budget et revenus")
        axis.set_xlabel("Budget")
        axis.set_ylabel("Revenus")
        plt.tight_layout()
        plt.show()
        return figure

    def plot_votes_popularity(self):
        figure, axis = plt.subplots()
        axis.scatter(self.data["vote_count"], self.data["popularity"])
        axis.set_title("Votes et popularite")
        axis.set_xlabel("Nombre de votes")
        axis.set_ylabel("Popularite")
        plt.tight_layout()
        plt.show()
        return figure

    def plot_boxplots(self):
        boxplot_data = self.data[["vote_average", "popularity", "runtime", "budget", "revenue"]].copy()
        boxplot_data[["budget", "revenue"]] = boxplot_data[["budget", "revenue"]].apply(lambda column: column.clip(upper=column.quantile(0.95)))
        figure, axis = plt.subplots()
        axis.boxplot(boxplot_data.values, labels=boxplot_data.columns)
        axis.set_title("Boxplots des variables numeriques")
        axis.set_ylabel("Valeur")
        axis.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        plt.show()
        return figure

    def plot_correlations(self):
        correlation_columns = ["vote_average", "popularity", "runtime", "budget", "revenue", "vote_count"]
        correlation = self.data[correlation_columns].corr()
        figure, axis = plt.subplots()
        image = axis.imshow(correlation, cmap="coolwarm")
        axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
        axis.set_yticks(range(len(correlation.columns)), correlation.columns)
        axis.set_title("Matrice de correlation")
        figure.colorbar(image)
        plt.tight_layout()
        plt.show()
        return figure
