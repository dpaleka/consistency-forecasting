import requests
from bs4 import BeautifulSoup

def dump_page_content(url, output_file):
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for bad responses

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Write the prettified HTML to a file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(soup.prettify())
        
        print(f"Page content dumped to {output_file}")
    except requests.RequestException as e:
        print(f"Request failed: {e}")

# URL to scrape
url = "https://www.metaculus.com/questions/349/spacex-lands-people-on-mars-by-2030/"

# Output file path
output_file = "metaculus_page_content.html"

# Execute the function
dump_page_content(url, output_file)