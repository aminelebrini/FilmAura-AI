from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import pandas as pd
class Kmeans:
    def __init__(
            self, 
            n_clusters: int = 5,
            max_iter: int = 300,
            random_state: int = None   
        ):
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state
        )
        self.preprocessor = None
        self.labels = None
        self.centroids = None

    def train(self, nums_cols , text_col , df: pd.DataFrame):

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), nums_cols),
                ("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), text_col)
            ]
        )

        X_processed = self.preprocessor.fit_transform(df)
        try:
            self.model.fit(X_processed)
            self.labels = self.model.labels_
        except Exception as e:
            print(f"Error during training: {e}")

        return self.labels

    def predict(self, X):
    
        try:
            X_processed = self.preprocessor.transform(X)
            return self.model.predict(X_processed)
        except Exception as e:
            print(f"Error during prediction: {e}")

    def get_centroids(self):

        try:
            self.centroids = self.model.cluster_centers_
        except Exception as e:
            print(f"Error retrieving centroids: {e}")

        return self.centroids
    

