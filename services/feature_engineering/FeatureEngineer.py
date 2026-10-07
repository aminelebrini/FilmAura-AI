import pandas as pd
class FeatureEngineer:
    def __init__(self):
        pass

    def feature_engineer_function(self, df : pd.DataFrame):

        data_frame = pd.DataFrame(df)

        data_frame['release_date'] = pd.to_datetime(
            data_frame['release_date'], errors='coerce'
        )

        data_frame['release_year'] = data_frame['release_date'].dt.year

        data_frame['release_month'] = data_frame['release_date'].dt.month

        data_frame['decade'] = (data_frame['release_year'] // 10) * 10
        
        data_frame['number_of_genres'] = data_frame['genres'].apply(
            lambda x: len(x) if isinstance(x, list) else 0  
        )
        
        data_frame['number_of_keywords'] = data_frame['keywords'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        data_frame['runtime_category'] = pd.cut(
            x=data_frame['runtime'],
            bins=[0, 90, 120, 180, float("inf")],
            labels=["Short", "Standard", "Long", "Epic"],
            right=False
        )

        data_frame['has_homepage'] = data_frame['homepage'].fillna('').astype(str).str.strip().ne('')

        data_frame['is_multilingual'] = data_frame['spoken_languages'].apply(
            lambda x: True if len(x) > 1 and isinstance(x, list) else False
        )

        return data_frame

        

