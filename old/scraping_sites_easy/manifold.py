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
    market_info = []
    for market in data:


        try:
            resolution_date = 'N/A'
            if market.get('closeTime'):
                timestamp = market['closeTime'] / 1000
                # Ensure the timestamp is within a reasonable range (e.g., after year 1970)
                if timestamp > 0:
                    resolution_date = datetime.fromtimestamp(timestamp).strftime('%B %d, %Y')
        except Exception as e:
            print(f"Error processing market {market['id']}: {e}")
            resolution_date = 'Error processing date'
        
        market_info.append({
            'id': market['id'],
            'title': market['question'],
            'body': market.get('description', 'N/A'),
            'question_type': market.get('outcomeType', '').lower(),
            'resolution_date': resolution_date,
            'url': market.get('url', '').lower(),
            'data_source': 'manifold',
            'metadata': {'topics': market.get('tags', [])},
            'resolution': market.get('isResolved', None)
        })
    
    return {
        "api_url": api_url,
        "markets": market_info
    }

if __name__ == "__main__":
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