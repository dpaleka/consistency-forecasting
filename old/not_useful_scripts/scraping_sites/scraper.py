import time
from request_handler import RequestHandler
from parser import Parser
from database import Database

class Scraper:
    def __init__(self, config_path='config.json', db_file='prediction_market.db'):
        """
        Initializes the Scraper with paths to the configuration and database files.
        :param config_path: Path to the configuration file.
        :param db_file: Path to the SQLite database file.
        """
        self.request_handler = RequestHandler(config_path)
        self.db = Database(db_file)

    def scrape_and_store(self):
        """
        Orchestrates the scraping of prediction market data, parsing of the data,
        and storing the parsed data into the database.
        """
        # Fetch market data using the RequestHandler
        raw_market_data = self.request_handler.get_market_data()
        
        if raw_market_data is None:
            print("No market data fetched. Exiting.")
            return
        
        # Parse the raw market data into structured data
        parser = Parser(raw_market_data)
        structured_data = parser.parse_data()
        
        if not structured_data:
            print("No data parsed. Exiting.")
            return
        
        # Ensure the prediction_market table exists
        self.db.create_prediction_market_table()
        
        # Insert parsed data into the database
        for data in structured_data:
            self.db.insert_prediction_market_data(data)
            print(f"Inserted data for market {data.market_id} into the database.")
        
        print("Scraping and storage process completed.")

if __name__ == "__main__":
    scraper = Scraper()
    scraper.scrape_and_store()
