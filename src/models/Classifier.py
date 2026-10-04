import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC , LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)


class Classifier:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline = None
        self.best_pipeline = None
        self.preprocessor = None
        self.best_model_name = ""
        self.result = {}
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "Linear SVM": LinearSVC(random_state=self.random_state)
        }

    def build_pipline(self, num_features: list, text_feature: str):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("text", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), text_feature)
            ]
        )

        return self.preprocessor

    def train(self, df: pd.DataFrame , target_col: str = "high_engagement", text_col: str = "overview"):
        data_frame = pd.DataFrame(df)

        X = data_frame.drop(columns=[target_col])
        Y = data_frame[target_col]

        nums_cols = (X.select_dtypes(include=["number"]).columns.drop([text_col], errors="ignore").to_list())

        self.build_pipline(num_features=nums_cols, text_feature=text_col)

        X_train, X_test, Y_train, Y_test = train_test_split(

            X, Y, test_size=0.2, random_state=self.random_state, stratify=Y
        )

        summary = []
        best_f1 = -1.0

        for name, classifier_model in self.models.items():

            pipeline = Pipeline(
                steps=[(
                    "preprocessor", self.preprocessor
                ),(
                    "classifier", classifier_model
                )]
            )

            pipeline.fit(X_train, Y_train)

            Y_predict = pipeline.predict(X_test)

            accuracy = accuracy_score(Y_test, Y_predict)
            precision = precision_score(Y_test, Y_predict, zero_division=0)
            recall = recall_score(Y_test, Y_predict, zero_division=0)
            f1 = f1_score(Y_test, Y_predict, zero_division=0)

            if hasattr(pipeline, "decision_function"):
                y_scores = pipeline.decision_function(X_test)
            else:
                y_scores = pipeline.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(Y_test, y_scores)

            Confusion_matrix = confusion_matrix(Y_test, Y_predict)


            self.result[name] = {
                "pipeline": pipeline,
                "accuracy":accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "confusion_matrix": Confusion_matrix,
            }

            summary.append({
                "model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
                "Confusion Matrix": Confusion_matrix,
            })

            if f1 > best_f1:
                best_f1 = f1
                self.pipeline = pipeline
                self.best_pipline = pipeline
                self.best_model_name = name

        results_df = pd.DataFrame(summary)

        return results_df


    def save_best_model(self, file_path: str):
        if self.best_pipline is not None:
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            joblib.dump(self.best_pipeline, file_path)
            print(f"Best model '{self.best_model_name}' saved successfully to"
            f" {file_path}"
            )
        else:
            print("No trained pipeline found to save!")



            
