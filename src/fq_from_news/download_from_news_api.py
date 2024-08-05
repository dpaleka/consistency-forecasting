import requests
import os
import math
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# Constants
NEWS_API_DATA_DUMP_DIR = "./data/news_feed_fq_generation/news_api/news_feed_data_dump"
NEWS_API_DOMAINS = [
    "apnews.com",
    "bloomberg.com",
    "reuters.com",
]
NEWS_API_NUM_ARTICLES_PER_PAGE = 100  # maximum allowed value
NEWS_API_DEFAULT_PARAMS = {
    "pageSize": NEWS_API_NUM_ARTICLES_PER_PAGE,
    "language": "en",
    "sortBy": "popularity",
    "domains": ",".join(NEWS_API_DOMAINS),
}
NEWS_API_ENDPOINT = "v2/everything"
NEWS_API_REQUEST_URL = f"https://newsapi.org/{NEWS_API_ENDPOINT}"


def parse_date(date_str: str) -> datetime:
    """
    Given a date in the format YYYY-MM-DD, returns the corresponding datestring.
    Has validation for the correct type.

    :date_str: The date as a string

    :returns: Given date as a datetime object
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise TypeError(
            f"Invalid date format: {date_str}. Date must be in YYYY-MM-DD format."
        )


def get_news_from_api(
    start_date: datetime, end_date: datetime, page_number: int, api_key: str
) -> dict:
    """
    Downloads popular news at the given page number from news API

    :start_date: Start date for downloading news
    :end_date: End date for downloading news
    :page_number: The page number to be downloaded
    :api_key: News API key

    :returns: dict containing the response
    """
    current_params = NEWS_API_DEFAULT_PARAMS.copy()
    current_params["from"] = start_date.strftime("%Y-%m-%d")
    current_params["to"] = end_date.strftime("%Y-%m-%d")
    current_params["page"] = page_number
    current_params["apiKey"] = api_key

    response = requests.get(NEWS_API_REQUEST_URL, params=current_params)
    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(
            f"Failed to retrieve data: {response.status_code}.\n\tError response Dict:\n\t\t{response.__dict__}"
        )


def _news_api_download_file_path(
    start_date: datetime, end_date: datetime, num_pages: int
) -> str:
    """
    File path to download the articles from News API

    :start_date: Start date for downloading news
    :end_date: End date for downloading news
    :num_pages: Number of pages (each containing max 100 articles) to be downloaded

    :returns: file path where the news will be downloaded
    """
    if num_pages == -1:
        news_save_file_name = f"news_api_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_all.jsonl"
    else:
        news_save_file_name = f"news_api_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}.jsonl"

    return os.path.join(NEWS_API_DATA_DUMP_DIR, news_save_file_name)


def _download_news_from_api_per_day(
    news_date: datetime, num_pages: int, api_key: str
) -> str:
    """
    Downloads popular news as response pages from News API from a set of curated domains

    :start_date: Start date for downloading news
    :end_date: End date for downloading news
    :num_pages: Number of pages (each containing max 100 articles) to be downloaded
    :api_key: News API key

    :returns: file where the data is downloaded
    """
    assert num_pages == -1 or num_pages > 0

    articles_download_path = _news_api_download_file_path(
        news_date, news_date, num_pages
    )

    # Make the data dump directory
    os.makedirs(NEWS_API_DATA_DUMP_DIR, exist_ok=True)

    # Failsafe to prevent redundant News API queries
    if os.path.exists(articles_download_path):
        print(
            f"\tThe data has already been downloaded to {articles_download_path}\n\tReturning without downloading anything new!"
        )
        return articles_download_path

    if not os.getenv("NEWS_API_KEY"):
        raise RuntimeError("OS enviroment for NEWS_API_KEY either missing or None!")

    first_page_news = get_news_from_api(news_date, news_date, 1, api_key)

    # Calculating number of pages requried
    total_number_of_pages = math.ceil(
        first_page_news["totalResults"] / NEWS_API_NUM_ARTICLES_PER_PAGE
    )
    num_pages = (
        total_number_of_pages
        if num_pages == -1 or num_pages > total_number_of_pages
        else num_pages
    )

    if num_pages > 1:
        print(
            f"\nIf you have a developer account API, you will not be able to access {num_pages} (>1) pages!\n"
        )

    print(
        f"Saving {num_pages} from the total available {total_number_of_pages} pages of"
        + f" news artilces with {NEWS_API_NUM_ARTICLES_PER_PAGE} articles each to {articles_download_path}"
    )

    # Saving the pages
    for page_number in range(1, num_pages + 1):
        if page_number == 1:
            news_articles = first_page_news["articles"]
        else:
            news_articles = get_news_from_api(
                news_date, news_date, page_number, api_key
            )["articles"]
        with open(articles_download_path, "a") as jsonl_file:
            for article in news_articles:
                jsonl_file.write(json.dumps(article) + "\n")

    return articles_download_path


def download_news_from_api(
    start_date: datetime, end_date: datetime, num_pages: int, api_key: str
) -> str:
    daily_article_download_paths = []
    init_date = start_date
    while init_date <= end_date:
        # download the individual articles for each day and save their paths
        daily_article_download_paths.append(
            _download_news_from_api_per_day(init_date, num_pages, api_key)
        )

        init_date += timedelta(days=1)
