import os


class MongoDBConnection:
    def __init__(self, host , port ,user, password, db_name):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name

    def connet(self):
