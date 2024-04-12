import requests
from bs4 import BeautifulSoup

def scrape_resolution_criteria(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Attempt to find the section containing "Resolution Criteria" by searching for a specific string.

        # Example of navigating to a sibling or child, assuming a consistent structure
        for criteria in soup.find_all(lambda tag: tag.name == "react-resolution-criteria"):
            next_sibling = criteria.find_next_sibling()  # If the text is in the next sibling
            if next_sibling:
                print(next_sibling.text)  # Print the text of the sibling

            child_paragraph = criteria.find('p')  # If the text is in a child paragraph
            if child_paragraph:
                print(child_paragraph.text)  # Print the text of the child paragraph
    except requests.RequestException as e:
        print(f"Request failed: {e}")

# Replace the URL with the actual page you want to scrape
url = "https://www.metaculus.com/questions/349/spacex-lands-people-on-mars-by-2030/"
scrape_resolution_criteria(url)