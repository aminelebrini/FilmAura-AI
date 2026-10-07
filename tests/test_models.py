import joblib
import pandas as pd

from models.Classifier import Classifier
from models.clustring.DBScan import DBScan
from models.clustring.Kmeans import Kmeans


def make_movies(rows=12):
    return pd.DataFrame(
        {
            "overview": [
                "hero saves the city" if index % 2 == 0 else "romantic family story"
                for index in range(rows)
            ],
            "popularity": [float(index + 1) for index in range(rows)],
            "vote_average": [6.0 + (index % 4) * 0.5 for index in range(rows)],
            "runtime": [90 + (index % 3) * 20 for index in range(rows)],
            "high_engagement": [index % 2 for index in range(rows)],
        }
    )


def test_classifier_trains_and_saves_model(tmp_path):
    classifier = Classifier(random_state=42)
    result = classifier.train(make_movies())
    model_path = tmp_path / "best_model.joblib"

    classifier.save_best_model(str(model_path))

    assert not result.empty
    assert classifier.best_pipline is not None
    assert model_path.exists()
    loaded_model = joblib.load(model_path)
    predictions = loaded_model.predict(make_movies().drop(columns=["high_engagement"]))
    assert len(predictions) == 12


def test_kmeans_trains_and_predicts():
    movies = make_movies()
    model = Kmeans(n_clusters=2, random_state=42)
    labels = model.train(["popularity", "vote_average", "runtime"], "overview", movies)

    predictions = model.predict(movies)

    assert len(labels) == len(movies)
    assert len(predictions) == len(movies)
    assert model.get_centroids().shape[0] == 2


def test_dbscan_trains_and_predicts():
    movies = make_movies()
    model = DBScan()
    labels = model.train(["popularity", "vote_average", "runtime"], "overview", movies)

    predictions = model.predict(movies)

    assert len(labels) == len(movies)
    assert len(predictions) == len(movies)
    assert model.get_core_samples() is not None
