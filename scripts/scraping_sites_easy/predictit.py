import requests
import json
from datetime import datetime

def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the webpage: {url}")
    
    data = response.json()

    market_info = []
    for market in data['markets']:
        # Extract the end date or set it to None if not available/applicable

        date_end = market['contracts'][0].get('dateEnd') if market['contracts'] else 'NA'
        if date_end != 'NA':
            try:
                resolution_date = datetime.strptime(date_end, "%Y-%m-%dT%H:%M:%SZ").strftime("%B %d, %Y")
            except ValueError:
                # Handle cases where date_end is not in the expected format
                resolution_date = "Unknown"
        else:
            resolution_date = "Unknown"

        market_dict = {
            'id': market.get('id'),
            'text': market.get('name'),
            'resolution_criteria': market.get('description', 'No description available'),
            'question_type': 'binary' if len(market.get('contracts', [])) == 1 else 'multiple_choice',
            'resolution_date': resolution_date,
            'url': market.get('url'),
            'data_source': 'predictit',
            'metadata': {
                'topics': market.get('tags', [])
            },
            'resolution': market.get('status'),  # Optionally, add logic to determine 'resolution' if available in the data
        }
        market_info.append(market_dict)
    
    return market_info

if __name__ == "__main__":
    url = "https://www.predictit.org/api/marketdata/all"
    
    try:
        # Scrape the website
        data = scrape_website(url)
        
        # Convert the data to JSON and print
        print(json.dumps(data, indent=4))

        with open('predictit.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

    except Exception as e:
        print(f"Error: {e}")