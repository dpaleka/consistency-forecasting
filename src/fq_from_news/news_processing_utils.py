import requests
import os
import math
import json
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tqdm import tqdm
import spacy
from common.path_utils import get_src_path

load_dotenv()

# Constants
news_api_default_data_dir = os.path.join(
    get_src_path(), "data/news_feed_fq_generation/news_api/news_feed_data_dump"
)

os.makedirs(news_api_default_data_dir, exist_ok=True)

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


def _get_news_from_api(
    start_date: datetime, end_date: datetime, page_number: int, api_key: str
) -> dict:
    """
    Downloads popular news at the given page number from News API.

    Args:
        start_date (datetime): Start date for downloading news.
        end_date (datetime): End date for downloading news.
        page_number (int): The page number to be downloaded.
        api_key (str): News API key.

    Returns:
        dict: Dictionary containing the response from the News API.

    Raises:
        RuntimeError: If the API request fails.
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
    File path to download the articles from News API.

    Args:
        start_date (datetime): Start date for downloading news.
        end_date (datetime): End date for downloading news.
        num_pages (int): Number of pages (each containing max 100 articles) to be downloaded.

    Returns:
        str: File path where the news will be downloaded.
    """
    if num_pages == -1:
        news_save_file_name = f"news_api_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_all.jsonl"
    else:
        news_save_file_name = f"news_api_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}.jsonl"

    return os.path.join(news_api_default_data_dir, news_save_file_name)


def _news_api_download_consolidation_path(
    start_date: datetime, end_date: datetime, num_pages: int
) -> str:
    """
    File path where the consolidated news of the per-day news will be saved.

    Args:
        start_date (datetime): Start date for downloading news.
        end_date (datetime): End date for downloading news.
        num_pages (int): Number of pages (each containing max 100 articles) to be downloaded.

    Returns:
        str: File path where the consolidated news will be saved.
    """
    if num_pages == -1:
        news_save_file_name = f"consolidated_news_api_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_all.jsonl"
    else:
        news_save_file_name = f"consolidated_news_api_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}.jsonl"

    return os.path.join(news_api_default_data_dir, news_save_file_name)


def _news_api_processed_news_path(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    processsed_news_save_directory,
) -> str:
    """
    File path where the processed form of the consolidated news will be saved.

    Args:
        start_date (datetime): Start date for downloading news.
        end_date (datetime): End date for downloading news.
        num_pages (int): Number of pages (each containing max 100 articles) to be downloaded.
        processsed_news_save_directory (str): Directory where to store consolidated news. If empty, defaults.

    Returns:
        str: File path where the processed news will be saved.
    """
    if (
        processsed_news_save_directory is None
        or len(processsed_news_save_directory.strip()) == 0
    ):
        save_dir = news_api_default_data_dir
    else:
        save_dir = processsed_news_save_directory
    os.makedirs(save_dir, exist_ok=True)

    if num_pages == -1:
        news_save_file_name = f"processed_news_api_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_all.jsonl"
    else:
        news_save_file_name = f"processed_news_api_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}.jsonl"

    return os.path.join(save_dir, news_save_file_name)


def _download_news_from_api_per_day(
    news_date: datetime, num_pages: int, api_key: str
) -> str:
    """
    Downloads popular news as response pages from News API from a set of curated domains.

    Args:
        news_date (datetime): Date for downloading news.
        num_pages (int): Number of pages (each containing max 100 articles) to be downloaded.
        api_key (str): News API key.

    Returns:
        str: File path where the data is downloaded.

    Raises:
        RuntimeError: If the News API key is not set in the environment.
    """
    assert num_pages == -1 or num_pages > 0

    articles_download_path = _news_api_download_file_path(
        news_date, news_date, num_pages
    )

    # Failsafe to prevent redundant News API queries
    if os.path.exists(articles_download_path):
        print(
            f"\tThe data has already been downloaded to {articles_download_path}\n\tReturning without downloading anything new!"
        )
        return articles_download_path

    if not os.getenv("NEWS_API_KEY"):
        raise RuntimeError("OS environment for NEWS_API_KEY either missing or None!")

    first_page_news = _get_news_from_api(news_date, news_date, 1, api_key)

    # Calculating number of pages required
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
        + f" news articles with {NEWS_API_NUM_ARTICLES_PER_PAGE} articles each to {articles_download_path}"
    )

    # Saving the pages
    for page_number in range(1, num_pages + 1):
        if page_number == 1:
            news_articles = first_page_news["articles"]
        else:
            news_articles = _get_news_from_api(
                news_date, news_date, page_number, api_key
            )["articles"]
        with open(articles_download_path, "a") as jsonl_file:
            for article in news_articles:
                jsonl_file.write(json.dumps(article) + "\n")

    return articles_download_path


def download_news_from_api(
    start_date: datetime, end_date: datetime, num_pages: int, api_key: str
) -> str:
    """
    Downloads and consolidates news articles from News API for a date range.

    Args:
        start_date (datetime): Start date for downloading news.
        end_date (datetime): End date for downloading news.
        num_pages (int): Number of pages (each containing max 100 articles) to be downloaded.
        api_key (str): News API key.

    Returns:
        str: File path where the consolidated news will be saved.
    """
    consolidated_news_path = _news_api_download_consolidation_path(
        start_date, end_date, num_pages
    )
    if os.path.exists(consolidated_news_path):
        print(
            f"The consolidated news has already been downloaded to {consolidated_news_path}. "
        )
        return consolidated_news_path

    daily_article_download_paths = []
    init_date = start_date
    while init_date <= end_date:
        # download the individual articles for each day and save their paths
        daily_article_download_paths.append(
            _download_news_from_api_per_day(init_date, num_pages, api_key)
        )

        init_date += timedelta(days=1)

    with open(consolidated_news_path, "w") as outfile:
        for file_path in daily_article_download_paths:
            with open(file_path, "r") as infile:
                for line in infile:
                    outfile.write(line)

    print(f"Saved the consolidated news to {consolidated_news_path}.")
    return consolidated_news_path


def _ner_news_processing(
    sorted_news_articles: list[dict], ner_threshold: float
) -> list[dict]:
    """
    Processes the downloaded news to remove repetitions using NER

    Args:
        sorted_news_articles (list): sorted news articles
        ner_threshold (float): The threshold used to designate an article as a duplicate

    Returns:
        list: Processed news without duplicates
    """
    # Load the SpaCy model
    ner_model = spacy.load("en_core_web_sm")

    def extract_named_entities(text):
        doc = ner_model(text)
        entities = set()
        for ent in doc.ents:
            entities.add((ent.text, ent.label_))
        return entities

    def are_articles_duplicates(entities1, entities2, threshold):
        intersection = entities1.intersection(entities2)
        similarity = len(intersection) / (
            len(entities1) + len(entities2) - len(intersection)
        )
        return similarity >= threshold

    def get_entities(article):
        entities_content, entities_description = [], []
        if article.get("content") is not None:
            entities_content = extract_named_entities(article.get("content"))
        if article.get("description") is not None:
            entities_description = extract_named_entities(article.get("description"))
        return entities_content.union(entities_description)

    entities_lst = []
    for article in tqdm(sorted_news_articles):
        entities_lst.append(get_entities(article))

    duplicates = set()
    n = len(sorted_news_articles)
    for i in tqdm(range(n)):
        if i in duplicates:
            continue
        entities_i = entities_lst[i]

        for j in range(i + 1, n):
            if j in duplicates:
                continue
            entities_j = entities_lst[j]
            if are_articles_duplicates(entities_i, entities_j, ner_threshold):
                duplicates.add(j)

    # Remove articles that are duplicates
    unique_articles = [sorted_news_articles[i] for i in range(n) if i not in duplicates]
    return unique_articles


def process_news(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    news_new_threshold: float,
    processsed_news_save_directory: str,
) -> str:
    """
    Processes the downloaded news to remove repetitions
        Fixes cases where the resolution changes with the advent of a more recent news.

    Args:
        start_date (datetime): Start date for downloading news.
        end_date (datetime): End date for downloading news.
        num_pages (int): Number of pages (each containing max 100 articles) to be downloaded.
        news_new_threshold (float): The threshold used to designate an article as a duplicate
        processsed_news_save_directory (str): directory where to store the processed news. If left empty, defaults.

    Returns:
        str: File path where the processed news is saved.
    """
    # Consolidated news articles need to have been downloaded
    consolidated_news_path = _news_api_download_consolidation_path(
        start_date, end_date, num_pages
    )
    if not os.path.exists(consolidated_news_path):
        raise RuntimeError(
            "Consolidated news does not exist at {consolidated_news_path}!"
        )

    processed_news_path = _news_api_processed_news_path(
        start_date, end_date, num_pages, processsed_news_save_directory
    )
    if os.path.exists(processed_news_path):
        print(f"The processed news has already been saved to {processed_news_path}. ")
        return processed_news_path

    def read_jsonl(file_path):
        with open(file_path, "r") as file:
            for line in file:
                yield json.loads(line)

    def sort_articles_by_date(articles):
        return sorted(
            articles,
            key=lambda x: datetime.fromisoformat(
                x["publishedAt"].replace("Z", "+00:00")
            ),
            reverse=True,
        )

    def load_and_sort_articles(file_path):
        articles = list(read_jsonl(file_path))
        sorted_articles = sort_articles_by_date(articles)
        return sorted_articles

    # Load and sort all articles in descending order of publish date
    sorted_news_articles = load_and_sort_articles(consolidated_news_path)

    ner_processed_news_articles = _ner_news_processing(
        sorted_news_articles, news_new_threshold
    )

    # from pprint import pprint; pprint(ner_processed_news_articles)

    with open(processed_news_path, "w") as outfile:
        for news_article in ner_processed_news_articles:
            outfile.write(json.dumps(news_article) + "\n")

    print(f"Saved the processed news to {processed_news_path}.")
    return processed_news_path
