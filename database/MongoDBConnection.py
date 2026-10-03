import os
from pymongo import MongoClient
from pymongo.errors import PyMongoError

class MongoDBConnection:
    def __init__(self, host , port, db_name):
        self.host = host
        self.port = port
        # self.user = user
        # self.password = password
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        uri = f"mongodb://{self.host}:{self.port}/{self.db_name}"
        try:
            self.client = MongoClient(uri)
            self.db = self.client[self.db_name]
            print("Connected to MongoDB")
        except PyMongoError as e:
            print(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None

    def get_collection(self, collection_name: str):
        if self.db is None:
            raise ConnectionError(
                "MongoDB connection failed. Check your MongoDB environment variables "
                "and Atlas network access."
            )
        return self.db[collection_name]

    def close(self):
        if self.client:
            self.client.close()
            print("Connection to MongoDB closed.")
