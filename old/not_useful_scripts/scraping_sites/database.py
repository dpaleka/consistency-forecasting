import sqlite3
from sqlite3 import Error

class Database:
    def __init__(self, db_file):
        """
        Initializes a Database instance, trying to connect to the specified SQLite database file.
        :param db_file: Path to the SQLite database file.
        """
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
            print("Connection to SQLite DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")

    def execute_query(self, query, params=()):
        """
        Executes a given SQL query on the connected database.
        :param query: A string containing the SQL query to be executed.
        :param params: A tuple of parameters to be bound to the query.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            self.conn.commit()
            print("Query executed successfully")
        except Error as e:
            print(f"The error '{e}' occurred")

    def execute_read_query(self, query, params=()):
        """
        Executes a read query and returns the fetched data.
        :param query: A string containing the SQL query to be executed.
        :param params: A tuple of parameters to be bound to the query.
        :return: Fetched data from the database.
        """
        cursor = self.conn.cursor()
        result = None
        try:
            cursor.execute(query, params)
            result = cursor.fetchall()
            return result
        except Error as e:
            print(f"The error '{e}' occurred")
            return None

    def create_prediction_market_table(self):
        """
        Creates the prediction_market table in the database if it doesn't already exist.
        """
        query = """
        CREATE TABLE IF NOT EXISTS prediction_market (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            website_name TEXT NOT NULL,
            market_id TEXT NOT NULL,
            market_name TEXT NOT NULL,
            market_type TEXT NOT NULL,
            outcomes TEXT NOT NULL
        );
        """
        self.execute_query(query)

    def insert_prediction_market_data(self, data):
        """
        Inserts a new row into the prediction_market table.
        :param data: An instance of PredictionMarketData.
        """
        query = """
        INSERT INTO prediction_market (
            website_name, market_id, market_name, market_type, outcomes
        ) VALUES (?, ?, ?, ?, ?);
        """
        params = (data.website_name, data.market_id, data.market_name, data.market_type, ','.join(data.outcomes))
        self.execute_query(query, params)

# Example usage
if __name__ == "__main__":
    db = Database("prediction_market.db")
    db.create_prediction_market_table()
    # Example data insertion, assuming you have an instance `data` of PredictionMarketData
    # db.insert_prediction_market_data(data)
