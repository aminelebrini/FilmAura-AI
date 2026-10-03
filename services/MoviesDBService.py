from database.MongoDBConnection import MongoDBConnection
from pymongo import UpdateOne
from pymongo.errors import PyMongoError

import pandas as pd

class MongoDBService:

    def __init__(self, df_conn: MongoDBConnection, collection_name):
        self.df_conn = df_conn
        self.collection_name = collection_name
        self.collection = self.df_conn.get_collection(collection_name)

    def load_data_to_db(self, df: pd.DataFrame):

        data_frame = df.copy()

        if data_frame.empty:
            print("DataFrame is empty. Nothing to load.")
            return 0

        for col in data_frame.select_dtypes(include=["datetime64"]).columns:
            data_frame[col] = data_frame[col].dt.strftime("%Y-%m-%d")

        records = data_frame.to_dict(orient="records")
        operation = []
        for doc in records:
            if "id" in doc:
                operation.append(
                    UpdateOne({
                        "id": doc["id"]
                        },{"$set": doc}, upsert=True)
                )
        try:
            result = self.collection.bulk_write(operation)
            return result
        except PyMongoError as e:
            print(f"Failed to load data to MongoDB: {e}")
            return 0