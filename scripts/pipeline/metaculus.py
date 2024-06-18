import requests
import json
from bs4 import BeautifulSoup
import argparse


def fetch_resolution_criteria(question_url):
    """
    Fetches the Resolution Criteria from a Metaculus question page.

    :param question_url: URL of the Metaculus question.
    :return: The Resolution Criteria text or None if not found.
    """
    response = requests.get(question_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        # Assuming the Resolution Criteria is contained within a specific HTML element
        # You'll need to inspect the page and adjust the selector accordingly
        resolution_criteria_element = soup.select_one(
            "div.resolution-criteria"
        )  # Adjust this selector

        if resolution_criteria_element:
            return resolution_criteria_element.text.strip()

    return None


def fetch_live_questions_with_dates(
    api_url, start_date=None, end_date=None, num_questions=500
):
    """
    Fetches live questions and their resolution dates from the Metaculus API.
    :param api_url: Base URL of the Metaculus API.
    :param start_date: Start date in 'yyyymmdd' format.
    :param end_date: End date in 'yyyymmdd' format.
    :return: A list of tuples containing live question titles and their resolution dates.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    questions_info = []
    total_questions = num_questions  # Total number of questions you want to fetch
    page_size = (
        100  # Number of questions per page. Adjust based on API's maximum allowed limit
    )
    page = 1

    seen_ids = set()
    while len(questions_info) < total_questions:
        params = {
            "limit": page_size,
            "offset": (page - 1)
            * page_size,  # or 'offset': (page-1) * page_size if the API uses offset
            "resolve_time__gt": start_date,
            "resolve_time__lt": end_date,
        }
        response = requests.get(f"{api_url}/questions", headers=headers, params=params)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch the API: {api_url}")

        questions_data = response.json()

        if len(questions_data.get("results", [])) == 0:
            break

        for question in questions_data.get("results", []):
            question_type = question.get("possibilities", {}).get("type")

            # Filter out non-binary questions
            if question_type != "binary":
                continue
            question_info = {
                "id": question.get("id"),
                "title": question.get("title"),
                "body": question.get(
                    "description"
                ),  # Assuming 'description' is the detailed text
                "question_type": question_type,
                "resolution_date": question.get(
                    "resolve_time"
                ),  # You might need to format this date
                "url": f"https://www.metaculus.com/questions/{question.get('id')}",
                "data_source": "metaculus",
                "metadata": {
                    "topics": question.get(
                        "tags", []
                    )  # Assuming 'tags' can be used as topics
                },
                "resolution": question.get("resolution"),
            }

            if (question_info["id"] not in seen_ids) and (
                (question_info["resolution"] is None)
                or (round(float(question_info["resolution"])) != -2)
            ):
                questions_info.append(question_info)
                seen_ids.add(question_info["id"])
                # print(len(questions_info), 'total_qs')
                # print(page, 'page')
                # print(question_info['title'])
                # print(question_info['resolution_date'])
                # print(question_info['url'])
                # print(question_info['resolution'])
            if len(questions_info) >= total_questions:
                break

        page += 1

    return questions_info[
        :total_questions
    ]  # Ensure only the requested number of questions are returned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch live questions from Metaculus API."
    )
    parser.add_argument("-start", type=str, help="Start date in yyyymmdd format.")
    parser.add_argument("-end", type=str, help="End date in yyyymmdd format.")
    parser.add_argument("-num", type=int, help="Number of questions to fetch.")
    args = parser.parse_args()

    api_url = "https://www.metaculus.com/api2"

    try:
        # Scrape the website
        data = fetch_live_questions_with_dates(api_url, args.start, args.end, args.num)

        # Convert the data to JSON and print
        # print(json.dumps(data, indent=4))
        print("total entries:", len(data))

        if args.start or args.end:
            with open(
                "metaculus_{}_{}.json".format(args.start, args.end), "w"
            ) as json_file:
                json.dump(data, json_file, indent=4)
        else:
            with open("metaculus.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

    except Exception as e:
        print(f"Error: {e}")


"""
def fetch_live_questions_with_dates(api_url):

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

    questions_info = []
    for question in questions_data.get('results', []):
        
        #div class = content prediction-section-resolution-criteria
        
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

    return questions_info

"""
