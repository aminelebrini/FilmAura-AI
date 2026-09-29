import pandas as pd
class DataCleaning:
    def __init__(self):
        pass

    def clean_data(self, films_data):

        cleaned_data = []
        data_df = pd.DataFrame(films_data)
        print(type(data_df))
        print(data_df.head())
        print(data_df.columns)
        print(data_df.info())
        print(data_df.describe())
        
        data_df["backdrop_path"] = data_df["backdrop_path"].fillna("/default-backdrop.jpg")
        
        if "id" in data_df.columns:
            data_df = data_df.drop_duplicates(subset=["id"])
            
        print(data_df.isnull().sum())
        # print(type(data_df["release_date"]))
        data_df["release_date"] = pd.to_datetime(data_df["release_date"], errors="coerce")
        #drop null date
        data_df = data_df.dropna(subset=["release_date"])

        #separation of columns between numeric and categories and texts
        nums_col = data_df.select_dtypes(include=["number"]).columns
        cats_cols = data_df.select_dtypes(include=["object","string"]).columns

        data_df[cats_cols] = data_df[cats_cols].fillna("")
        data_df[nums_col] = data_df[nums_col].fillna(0)

        # print(type(data_df["release_date"]))
        print(data_df.isnull().sum())
        return data_df
       
        

        
