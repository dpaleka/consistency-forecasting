import pytest
from unittest.mock import Mock, patch
from forecasters.llm_forecasting.information_retrieval import (
    retrieve_gnews_articles_fulldata,
)
from gnews import GNews


@pytest.fixture
def mock_gnews():
    with patch("forecasters.llm_forecasting.information_retrieval.GNews") as mock:
        yield mock


@pytest.fixture
def sample_article():
    return {
        "title": "5 Trump family members who are involved in his 2024 campaign — and 5 who aren't - Business Insider",
        "description": "5 Trump family members who are involved in his 2024 campaign — and 5 who aren't  Business Insider",
        "published date": "Wed, 25 Oct 2023 07:00:00 GMT",
        "url": "https://news.google.com/rss/articles/CBMigAFBVV95cUxQYlUycXI0XzdsdWt6SHJDcUNIbFNnT1M3aVc5bnZEa2dUZnZFaWJmTWlsSGJHNVVTeGhqNGVZSmNfTlhLVnVFOUZHM2pLTjJxUWh3c19oMDFiXzZzT013QTRFNzdWbmtHRGdEVFdkX1c2MWI4czZLLUJOUUstVDBFaw?oc=5&hl=en-US&gl=US&ceid=US:en",
        "publisher": {
            "href": "https://www.businessinsider.com",
            "title": "Business Insider",
        },
        "search_term": "Trump family 2024 candidacy announcement",
    }


def test_get_full_article(sample_article):
    # Arrange
    gnews = GNews()

    # Act
    result = gnews.get_full_article(sample_article["url"])

    print(f"{result=}")
    print(f"{result.text_cleaned=}")

    # Assert
    assert result is not None
    assert result.text_cleaned is not None
    assert len(result.text_cleaned) > 0
    assert result.publish_date is not None
    assert result.html is not None


def test_retrieve_gnews_articles_fulldata_real(sample_article):
    # Arrange
    retrieved_articles = [[sample_article]]

    # Act
    result = retrieve_gnews_articles_fulldata(
        retrieved_articles, num_articles=1, length_threshold=200
    )

    # Assert
    assert len(result) == 1
    assert result[0].search_term == sample_article["search_term"]
    assert result[0].text_cleaned is not None
    assert len(result[0].text_cleaned) > 200
    assert result[0].publish_date is not None
    assert result[0].html == ""

    # Print some information about the retrieved article
    print(f"Article title: {result[0].title}")
    print(f"Article publish date: {result[0].publish_date}")
    print(f"Article text length: {len(result[0].text_cleaned)}")
    print(f"First 200 characters of article text: {result[0].text_cleaned[:200]}")


def test_retrieve_gnews_articles_fulldata(mock_gnews, sample_article):
    # Arrange
    mock_full_article = Mock(spec=GNews)
    mock_full_article.text_cleaned = "This is the cleaned text of the article."
    mock_full_article.publish_date = "2023-10-25"

    mock_gnews_instance = mock_gnews.return_value
    mock_gnews_instance.get_full_article.return_value = mock_full_article

    retrieved_articles = [[sample_article]]

    # Act
    result = retrieve_gnews_articles_fulldata(
        retrieved_articles, num_articles=1, length_threshold=10
    )

    # Assert
    assert len(result) == 1
    assert result[0].search_term == sample_article["search_term"]
    assert result[0].text_cleaned == mock_full_article.text_cleaned
    assert result[0].publish_date == mock_full_article.publish_date
    assert result[0].html == ""

    mock_gnews_instance.get_full_article.assert_called_once_with(sample_article["url"])


def test_retrieve_gnews_articles_fulldata_short_article(mock_gnews, sample_article):
    # Arrange
    mock_full_article = Mock(spec=GNews)
    mock_full_article.text_cleaned = "Short."
    mock_full_article.publish_date = "2023-10-25"

    mock_gnews_instance = mock_gnews.return_value
    mock_gnews_instance.get_full_article.return_value = mock_full_article

    retrieved_articles = [[sample_article]]

    # Act
    result = retrieve_gnews_articles_fulldata(
        retrieved_articles, num_articles=1, length_threshold=10
    )

    # Assert
    assert len(result) == 0
    mock_gnews_instance.get_full_article.assert_called_once_with(sample_article["url"])


def test_retrieve_gnews_articles_fulldata_no_publish_date(mock_gnews, sample_article):
    # Arrange
    mock_full_article = Mock(spec=GNews)
    mock_full_article.text_cleaned = "This is the cleaned text of the article."
    mock_full_article.publish_date = None

    mock_gnews_instance = mock_gnews.return_value
    mock_gnews_instance.get_full_article.return_value = mock_full_article

    retrieved_articles = [[sample_article]]

    # Act
    result = retrieve_gnews_articles_fulldata(
        retrieved_articles, num_articles=1, length_threshold=10
    )

    # Assert
    assert len(result) == 0
    mock_gnews_instance.get_full_article.assert_called_once_with(sample_article["url"])


def test_retrieve_gnews_articles_fulldata_duplicate_url(mock_gnews, sample_article):
    # Arrange
    mock_full_article = Mock(spec=GNews)
    mock_full_article.text_cleaned = "This is the cleaned text of the article."
    mock_full_article.publish_date = "2023-10-25"

    mock_gnews_instance = mock_gnews.return_value
    mock_gnews_instance.get_full_article.return_value = mock_full_article

    retrieved_articles = [[sample_article, sample_article]]

    # Act
    result = retrieve_gnews_articles_fulldata(
        retrieved_articles, num_articles=2, length_threshold=10
    )

    # Assert
    assert len(result) == 1
    mock_gnews_instance.get_full_article.assert_called_once_with(sample_article["url"])
