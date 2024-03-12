import requests
import json

def fetch_live_questions_with_dates(api_url):
    """
    Fetches live questions and their resolution dates from the Metaculus API.
    :param api_url: Base URL of the Metaculus API.
    :return: A list of tuples containing live question titles and their resolution dates.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    params = {
        'status': 'open',  # Adjust based on actual API documentation
    }
    response = requests.get(f"{api_url}/questions", headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the API: {api_url}")

    questions_data = response.json()

    
    # Assuming the response structure and that 'resolution_date' is the key for the resolution date
    # Adjust the key based on actual API response structure

        
    questions_info = [{
        'name': question['title'],
        'end_date': question.get('resolve_time'),
        'question_type': question.get('possibilities')['type'],
        'url': f"https://www.metaculus.com/questions/{question['id']}"  # Constructing the URL using question ID

    } for question in questions_data.get('results', [])]
        
        


    """
        market_info = [{
        'name': market['name'],
        'end_date': market['contracts'][0]['dateEnd']  # Use .get() to avoid KeyError if 'endDate' is missing
    } for market in data['markets']]
    """

    return questions_info
"""
    
    return {
        "api_url": api_url,
        "markets": questions_info
    }
"""

if __name__ == "__main__":
    api_url = "https://www.metaculus.com/api2"
    
    try:
        # Scrape the website
        data = fetch_live_questions_with_dates(api_url)
        
        # Convert the data to JSON and print
        print(json.dumps(data, indent=4))
        with open('metaculus.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

    except Exception as e:
        print(f"Error: {e}")