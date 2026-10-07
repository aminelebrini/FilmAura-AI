from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.sparse import issparse
import pandas as pd

class ClustringEvaluator:
    def __init__(self, models):
        self.models = models
        self.evaluation_best_model = None

    def evaluate(self, X):

        results = []
        for name, model in self.models.items():
            labels = model.labels
            X_processed = model.preprocessor.transform(X)
            if issparse(X_processed):
                X_processed = X_processed.toarray()

            unique_labels = set(labels) - {-1}
            scoring_labels = set(labels)
            if 2 <= len(scoring_labels) < len(labels):
                sillouette = silhouette_score(X_processed, labels=labels)
                calinski = calinski_harabasz_score(X_processed, labels=labels)
                davies = davies_bouldin_score(X_processed, labels=labels)
            else:
                sillouette = float("nan")
                calinski = float("nan")
                davies = float("nan")

            results.append({
                "model": name,
                "n_clusters": len(unique_labels),
                "silhouette_score": sillouette,
                "calinski_harabasz_score": calinski,
                "davies_bouldin_score": davies
            })

        results_df = pd.DataFrame(results)

        return results_df

