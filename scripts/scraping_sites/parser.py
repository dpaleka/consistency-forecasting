import json
from data_model import PredictionMarketData

class Parser:
    def __init__(self, market_data):
        """
        Initializes the Parser with market data.
        :param market_data: List of market data dictionaries from various prediction market websites.
        """
        self.market_data = market_data

    def parse_data(self):
        """
        Parses the raw market data into a structured format.
        :return: List of PredictionMarketData instances.
        """
        parsed_data = []
        for market in self.market_data:
            website_name = market['name']
            for entry in market['data']:
                try:
                    # Assuming the data structure from each market website is consistent and contains these fields.
                    # This will need to be adjusted based on the actual structure of the data.
                    market_id = entry['id']
                    market_name = entry['name']
                    market_type = entry['type']
                    outcomes = entry['outcomes'] if 'outcomes' in entry else []
                    parsed_entry = PredictionMarketData(
                        website_name=website_name,
                        market_id=market_id,
                        market_name=market_name,
                        market_type=market_type,
                        outcomes=outcomes
                    )
                    parsed_data.append(parsed_entry)
                except KeyError as e:
                    print(f"Missing expected key {e} in market data entry from {website_name}.")
        return parsed_data

if __name__ == "__main__":
    # Example usage
    from request_handler import RequestHandler
    handler = RequestHandler()
    raw_data = handler.get_market_data()
    parser = Parser(raw_data)
    structured_data = parser.parse_data()
    for data in structured_data:
        print(data)  # Assuming the __str__ method is defined in PredictionMarketData for meaningful output.
