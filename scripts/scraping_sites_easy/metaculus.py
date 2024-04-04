import requests
import json

import requests
from bs4 import BeautifulSoup



def fetch_resolution_criteria(question_url):
    """
    Fetches the Resolution Criteria from a Metaculus question page.
    
    :param question_url: URL of the Metaculus question.
    :return: The Resolution Criteria text or None if not found.
    """
    response = requests.get(question_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        
        # Assuming the Resolution Criteria is contained within a specific HTML element
        # You'll need to inspect the page and adjust the selector accordingly
        resolution_criteria_element = soup.select_one('div.resolution-criteria')  # Adjust this selector
        
        if resolution_criteria_element:
            return resolution_criteria_element.text.strip()
    
    return None

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
    
    """    
    questions_info = [{
        'name': question['title'],
        'end_date': question.get('resolve_time'),
        'question_type': question.get('possibilities')['type'],
        'url': f"https://www.metaculus.com/questions/{question['id']}"  # Constructing the URL using question ID

    } for question in questions_data.get('results', [])]
        
    """

    questions_info = []
    for question in questions_data.get('results', []):
        
        #div class = content prediction-section-resolution-criteria
        
        """
        url = f"https://www.metaculus.com/questions/{question.get('id')}"
        print(url)
        resolution_criteria = fetch_resolution_criteria(url)
        print(resolution_criteria)
        """

        question_info = {
            'id': question.get('id'),
            'title': question.get('title'),
            'body': question.get('description'),  # Assuming 'description' is the detailed text
            'question_type': question.get('possibilities', {}).get('type'),
            'resolution_date': question.get('resolve_time'),  # You might need to format this date
            'url': f"https://www.metaculus.com/questions/{question.get('id')}",
            'data_source': 'metaculus',  # Assuming all questions are from Metaculus as per your context
            'metadata': {
                'topics': question.get('tags', [])  # Assuming 'tags' can be used as topics
            },
            'resolution': question.get('resolution')  # Assuming there's a 'resolution' field for resolved questions
        }
        questions_info.append(question_info)


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