from sklearn.model_selection import StratifiedKFold , cross_validate , GridSearchCV
import numpy as np
import pandas as pd
import os
import joblib
class Evaluator:

    def __init__(self, model):
        self.model = model

        self.grid_search_result = None

    def evaluate(self, X, y , cv: int = 5):

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }

        scores = cross_validate(
            self.model,
            X,
            y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1
            #if n_jobs take -1 he use all availble cpu cores 
        )

        summary = {}
        result = None
        for metric in scoring.keys():
            summary[metric] = {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"]))
            }
        result = pd.DataFrame(summary)

        return result

    def best_params_by_grid_searchcv(self, X, y , params_grid_search: dict , scoring: str = "f1", save_path : str = None):

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=params_grid_search,
            scoring=scoring,
            n_jobs=-1
        )

        grid_search.fit(X,y)

        grid_search_result = grid_search

        self.grid_search_result = grid_search_result

        print(f"Best Parameters ({scoring}): {grid_search.best_params_} !")
        print(f"Best CV Score ({scoring}): {grid_search.best_score_:.4f} !")

        if save_path:
            dir_name = os.path.dirname(save_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                joblib.dump(grid_search.best_estimator_, save_path)
            print(f"Optimized model saved to {save_path} !")

        return grid_search.best_estimator_ 




        




            