from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List
import pandas as pd
import numpy as np

class TextFeatureExtractor:
    def __init__(self,
            max_features: int = 5000,
            ngram_range: Tuple[int, int] = (1,2),
            stop_words: str = "english",
        ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words
        )
        self.features_name: List[str] = []
        
        self.configuration = [
            {"max_features": 1000, "ngram_range": (1, 1)},
            {"max_features": 3000, "ngram_range": (1, 1)},
            {"max_features": 3000, "ngram_range": (1, 2)},
            {"max_features": 5000, "ngram_range": (1, 2)},
        ]

    def transform_overview(self, df: pd.DataFrame, text_column = "overview"):

        data_frame = pd.DataFrame(df)

        text_clenead = data_frame[text_column].fillna("").astype(str)

        tfidf_matrix = self.vectorizer.fit_transform(text_clenead)

        self.features_name = self.vectorizer.get_feature_names_out().tolist()

        return tfidf_matrix

    def execute_experimentation(self, df: pd.DataFrame, text_column: str = "overview"):

        data_frame = pd.DataFrame(df)
        text_clenead = data_frame[text_column].fillna("").astype(str)

        results = []
        for configs in self.configuration:

            vec = TfidfVectorizer(
                max_features=configs["max_features"],
                ngram_range=configs["ngram_range"],
                stop_words="english"
            )
            matrix = vec.fit_transform(text_clenead)
            name_of_features = vec.get_feature_names_out(matrix)

            sparsity = (1.0 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100

            mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
            top_indices = mean_scores.argsort()[::-1][:3]
            top_terms = []
            for indice in top_indices:
                top_terms.append(name_of_features[indice])

            results.append({
                "ngram_range": str(configs["ngram_range"]),
                "max_features": configs["max_features"],
                "Shape": f"{matrix.shape[0]} x {matrix.shape[1]}",
                "Sparsity (%)": f"{sparsity:.2f}%",
                "Top 3 Terms" : ", ".join(top_terms)
            })

        results = pd.DataFrame(results)
        return results

        # return mean_scores, top_indices

        

