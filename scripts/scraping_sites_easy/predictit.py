import requests
import json
from datetime import datetime

def scrape_website(url):
    """
    Scrapes the PredictIt API at the given URL and extracts the 'Name' field and 'End Date' from each market.
    :param url: URL of the PredictIt API.
    :return: A dictionary containing the extracted names and their estimated resolution dates.
    """
    # Fetch the content from the URL
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the webpage: {url}")
    
    # Parse the JSON content
    data = response.json()
    #print(data)
    # Extract 'Name' and 'End Date' from each market
    market_info = [{
        'name': market['name'],
        'end_date': market['contracts'][0]['dateEnd']  # Use .get() to avoid KeyError if 'endDate' is missing
    } for market in data['markets']]
    
    # Return the names and end dates as a dictionary (customize keys based on your needs)
    return {
        "api_url": url,
        "markets": market_info
    }

if __name__ == "__main__":
    url = "https://www.predictit.org/api/marketdata/all"
    
    try:
        # Scrape the website
        data = scrape_website(url)
        
        # Convert the data to JSON and print
        print(json.dumps(data, indent=4))
    except Exception as e:
        print(f"Error: {e}")