import requests
import json

class RequestHandler:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Load the configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Configuration file {self.config_path} not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the configuration file {self.config_path}.")
            return None

    def get_market_data(self):
        """Fetch market data from all configured prediction market websites."""
        if not self.config:
            print("Configuration not loaded. Cannot proceed with data fetching.")
            return None

        market_data = []
        for website in self.config['prediction_market_websites']:
            url = website['base_url'] + website['markets_endpoint']
            headers = website.get('headers', {})
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    market_data.append({
                        'name': website['name'],
                        'data': response.json()
                    })
                else:
                    print(f"Failed to fetch data from {website['name']}. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Request to {website['name']} failed due to an exception: {e}")
        return market_data

if __name__ == "__main__":
    handler = RequestHandler()
    data = handler.get_market_data()
    print(data)
