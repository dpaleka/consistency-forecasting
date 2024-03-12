import requests
import json
from datetime import datetime

def scrape_manifold_markets(api_url):
    """
    Fetches market data from the Manifold Markets API and extracts relevant information.
    :param api_url: URL of the Manifold Markets API endpoint.
    :return: A dictionary containing the extracted market names, their URLs, and other relevant data.
    """
    # Fetch the content from the API URL
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the API data: {api_url}")
    
    # Parse the JSON content
    data = response.json()
    
    # Extract relevant data from each market
    market_info = [{
        'name': market['question'],
        'end_date': datetime.fromtimestamp(market['closeTime'] / 1000).strftime('%Y-%m-%d %H:%M:%S') if market.get('closeTime') else 'N/A',
        'question_type': market.get('outcomeType').lower(),
        'url':market.get('url').lower(),
    } for market in data]
    
    return market_info
"""
    return {
        "api_url": api_url,
        "markets": market_info
    }
"""
if __name__ == "__main__":
    #api_url = "https://api.manifold.markets/v0/markets?limit=10"
    api_url = "https://api.manifold.markets/v0/markets"
    
    try:
        # Scrape the Manifold Markets
        data = scrape_manifold_markets(api_url)
        
        # Convert the data to JSON and print
        print(json.dumps(data, indent=4))
        with open('manifold.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

    except Exception as e:
        print(f"Error: {e}")