import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

from database.MongoDBConnection import MongoDBConnection
from models.Classifier import Classifier
from services.MoviesDBService import MongoDBService
from services.feature_engineering.FeatureEngineer import FeatureEngineer
from src.DataCleaning import DataCleaning
from src.DataExtractor import DataExtractor
from src.DataLoad import DataLoad

ROOT = Path(__file__).resolve().parents[1]
BRONZE_PATH = ROOT / "data" / "bronze" / "data_bronze.json"
SILVER_PATH = ROOT / "data" / "silver" / "data_silver.json"
GOLD_PATH = ROOT / "data" / "gold" / "data_gold.json"
MODEL_PATH = ROOT / "models" / "best_model" / "best_model.joblib"


def extract() -> None:
    movies = DataExtractor().extract_all_data(int(os.getenv("TMDB_PAGE", "1")))
    DataLoad().load_data(movies, BRONZE_PATH)


def clean() -> None:
    with BRONZE_PATH.open(encoding="utf-8") as file:
        films = json.load(file)
    cleaned = DataCleaning().clean_data(films)
    DataLoad().load_data(cleaned.assign(release_date=cleaned["release_date"].dt.strftime("%Y-%m-%d")).to_dict("records"), SILVER_PATH)


def features() -> None:
    with SILVER_PATH.open(encoding="utf-8") as file:
        films = json.load(file)
    engineered = FeatureEngineer().feature_engineer_function(films)
    engineered["release_date"] = engineered["release_date"].dt.strftime("%Y-%m-%d")
    DataLoad().load_data(engineered.to_dict("records"), GOLD_PATH)


def load_mongodb() -> None:
    with GOLD_PATH.open(encoding="utf-8") as file:
        films = json.load(file)
    connection = MongoDBConnection(
        os.getenv("MONGO_HOST", "localhost"),
        os.getenv("MONGO_PORT", "27017"),
        os.getenv("MONGO_DB_NAME", "filmaura_db"),
    )
    connection.connect()
    try:
        MongoDBService(connection, os.getenv("MONGO_COLLECTION", "movies")).load_data_to_db(pd.DataFrame(films))
    finally:
        connection.close()


def train_model() -> None:
    with GOLD_PATH.open(encoding="utf-8") as file:
        films = pd.DataFrame(json.load(file))
    threshold = films["vote_count"].quantile(0.75)
    films["high_engagement"] = (films["vote_count"] > threshold).astype(int)
    Classifier(random_state=42).train(films, target_col="high_engagement", text_col="overview")
    classifier = Classifier(random_state=42)
    classifier.train(films, target_col="high_engagement", text_col="overview")
    classifier.save_best_model(str(MODEL_PATH))


with DAG(
    dag_id="film_aura_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["filmaura"],
) as dag:
    extraction = PythonOperator(task_id="extraction", python_callable=extract)
    nettoyage = PythonOperator(task_id="nettoyage", python_callable=clean)
    feature_engineering = PythonOperator(task_id="features", python_callable=features)
    mongodb = PythonOperator(task_id="mongodb", python_callable=load_mongodb)
    ml = PythonOperator(task_id="ml", python_callable=train_model)

    extraction >> nettoyage >> feature_engineering >> mongodb >> ml
