from src.DataExtractor import DataExtractor
from src.DataLoad import DataLoad
from src.DataCleaning import DataCleaning
from services.MoviesDBService import MongoDBService
from database.MongoDBConnection import MongoDBConnection
from notebooks.Visualisation import Visualisation
from services.feature_engineering.FeatureEngineer import FeatureEngineer
from services.feature_engineering.TextFeatureExtractor import TextFeatureExtractor
from models.Classifier import Classifier
from models.clustring.Kmeans import Kmeans
from models.clustring.DBScan import DBScan
from models.evaluation.Evaluator import Evaluator
from models.evaluation.clustring_evaluation.ClustringEvaluator import ClustringEvaluator
from pathlib import Path
from json import load , dump
import os
from dotenv import load_dotenv
from rich.console import Console

def main():
    console = Console()

    load_dotenv()
    data_load = DataLoad()
    data_extractor = DataExtractor()
    data_cleaning = DataCleaning()
    feature_engineer = FeatureEngineer()

    host = os.getenv("MONGO_HOST")
    port = os.getenv("MONGO_PORT")
    user = os.getenv("MONGO_USER")
    password = os.getenv("MONGO_PASS")
    db_name = os.getenv("MONGO_DB_NAME")

    BASE_DIR = Path(__file__).parent
    bronze_path = BASE_DIR / "data" / "bronze" / "data_bronze.json"
    silver_path = BASE_DIR / "data" / "silver" / "data_silver.json"
    gold_path = BASE_DIR / "data" / "gold" / "data_gold.json"
    # result = data_extractor.extract_all_data(100)
    # data_load.load_silver_data(result)
    with open(bronze_path, "r") as f:
            films_data = load(f)
    
    cleaned_data_to_silver = data_cleaning.clean_data(films_data)

    cleaned_data_to_silver["release_date"] = cleaned_data_to_silver["release_date"].dt.strftime("%Y-%m-%d")
    cleaned_data = cleaned_data_to_silver.to_dict(orient="records")
    print(cleaned_data)
    # with open(silver_path, "w", encoding="utf-8") as f:
    #     dump(cleaned_data, f, ensure_ascii=False, indent=4)

    # db_connection = MongoDBConnection(host, port, db_name)
    # db_connection.connect()
    # db_service = MongoDBService(db_connection, "movies")
    # db_service.load_data_to_db(cleaned_data_to_silver)
    
    

    print("="*50)
    print("             visualisation           ")
    print("="*50)
    data_visualization = Visualisation(cleaned_data_to_silver)
    data_visualization.plot_ratings_popularity()
    data_visualization.plot_genres()
    data_visualization.plot_releases()
    data_visualization.plot_runtime()
    data_visualization.plot_finance()
    data_visualization.plot_votes_popularity()

    print("="*50)
    print("             fin visualisation           ")
    print("="*50)


    with open(silver_path, "r") as file:
        data_json = load(file)


    with console.status("[ feature engineering in progress...") as status:
        new_data = feature_engineer.feature_engineer_function(data_json)

    new_data["release_date"] = new_data["release_date"].dt.strftime("%Y-%m-%d")
    new_data_json = new_data.to_dict(orient="records")
    data_load.load_data(new_data_json, gold_path)


    print("="*50)
    print("         TF-IDF          ")
    print("="*50)

    extractor = TextFeatureExtractor(max_features=1000, ngram_range=(1,1))
    Transform_overview = extractor.transform_overview(new_data_json, "overview")
    experimentation = extractor.execute_experimentation(new_data_json, "overview")

    print(Transform_overview)
    print(experimentation)

    print("="*50)
    print("         FIN TF-IDF          ")
    print("="*50)

    print("="*50)
    print("         Strating Classifier          ")
    print("="*50)

    threshold = new_data["vote_count"].quantile(0.75)
    new_data["high_engagement"] = (new_data["vote_count"] > threshold).astype(int)
    new_data = new_data.drop(columns=["vote_count"])
    classifier = Classifier(random_state=42)
    data = classifier.train(new_data, target_col="high_engagement", text_col="overview")

    # print(data)
    classifier.save_best_model("models/best_model/best_model.joblib")
    best_pipline = classifier.best_pipline
    print("="*50)
    print("         END Classifier          ")
    print("="*50)

    print("="*50)
    print("         Starting Evaluation          ")
    print("="*50)

    params_gridSearch_cv_logistic_regression = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['saga'],
        'classifier__penalty': ['elasticnet'],
        'classifier__l1_ratio': [0.0, 0.5, 1.0],
        'classifier__max_iter': [1000]
    }
    X = new_data.drop(columns=["high_engagement", "vote_count"], errors="ignore")
    Y = new_data["high_engagement"]
    evaluator = Evaluator(best_pipline)
    evaluation = evaluator.evaluate(X=X,y=Y,cv=5)
    print(evaluation)
    final_evaluation_result = evaluator.best_params_by_grid_searchcv(X=X,y=Y,
        params_grid_search=params_gridSearch_cv_logistic_regression,scoring="f1", save_path="models/best_model/best_model.joblib")

    print(final_evaluation_result)

    print("="*50)
    print("         End Evaluation          ")
    print("="*50)

    print("="*50)
    print("         Starting Clustering          ")
    print("="*50)

    print("="*50)
    print("         Starting KMeans Clustering          ")
    print("="*50)
    nums_cols = (new_data.select_dtypes(include=["number"]).columns.drop(["high_engagement", "vote_count"], errors="ignore").to_list())
    text_col = "overview"
    kmeans = Kmeans(n_clusters=5, max_iter=300, random_state=42)
    labels = kmeans.train(nums_cols=nums_cols, text_col=text_col, df=new_data)
    predictions = kmeans.predict(new_data.drop(columns=["high_engagement", "vote_count"], errors="ignore"))
    centroids = kmeans.get_centroids()
    print("Clustering Labels:", labels)
    print("Clustering Predictions:", predictions)
    print("Clustering Centroids:", centroids)
    print("="*50)
    print("         End KMeans Clustering          ")   
    print("="*50)

    print("="*50)
    print("         Starting DBScan Clustering          ")
    print("="*50)
    dbscan = DBScan()
    dbscan_labels = dbscan.train(nums_cols=nums_cols, text_col=text_col, df=new_data)
    dbscan_predictions = dbscan.predict(new_data.drop(columns=["high_engagement", "vote_count"], errors="ignore"))
    dbscan_core_samples = dbscan.get_core_samples()
    print("DBScan Clustering Labels:", dbscan_labels)
    print("DBScan Clustering Predictions:", dbscan_predictions)
    print("DBScan Clustering Core Samples:", dbscan_core_samples)

    print("="*50)
    print("         End DBScan Clustering          ")
    print("="*50)

    print("="*50)
    print("         End Clustering          ")
    print("="*50)

    print("="*50)
    print("         Starting Evaluation of Clustering            ")
    print("="*50)

    models = {
        "KMeans": kmeans,
        "DBScan": dbscan
    }

    clustering_evaluator = ClustringEvaluator(models=models)
    clustering_evaluation_result = clustering_evaluator.evaluate(new_data.drop(columns=["high_engagement", "vote_count"], errors="ignore"))
    print(clustering_evaluation_result)
    print("="*50)
    print("         End Evaluation of Clustering            ")
    print("="*50)

    # print(new_data)
if __name__ == "__main__":
    main()