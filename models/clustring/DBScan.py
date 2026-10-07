from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
class DBScan:
    def __init__(self):

        self.model = DBSCAN()
        self.labels = None
        self.preprocessor = None

    def train(self, nums_cols, text_col , df: pd.DataFrame):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), nums_cols),
                ("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), text_col)
            ]
        )
        
        try:
            X_processed = self.preprocessor.fit_transform(df)
            self.model.fit(X_processed)
            self.labels = self.model.labels_
            return self.labels
        except Exception as e:
            print(f"Error during training: {e}")

    def predict(self, X):
        try:
            X_processed = self.preprocessor.transform(X)
            return self.model.fit_predict(X_processed)
        except Exception as e:
            print(f"Error during prediction: {e}")

    def get_core_samples(self):
        try:
            return self.model.core_sample_indices_
        except Exception as e:
            print(f"Error retrieving core samples: {e}")
            return None
            