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


from forecasters.llm_forecasting.utils import gnews_utils
from urllib.parse import urlparse


def test_gnews_url_decoding(sample_article):
    sample_urls = [
        (
            "https://news.google.com/rss/articles/CBMipgFBVV95cUxPWV9fTEI4cjh1RndwanpzNVliMUh6czg2X1RjeEN0YUctUmlZb0FyeV9oT3RWM1JrMGRodGtqTk1zV3pkNEpmdGNxc2lfd0c4LVpGVENvUDFMOEJqc0FCVVExSlRrQmI3TWZ2NUc4dy1EVXF4YnBLaGZ4cTFMQXFFM2JpanhDR3hoRmthUjVjdm1najZsaFh4a3lBbDladDZtVS1FMHFn?oc=5",
            "https://www.reuters.com/business/healthcare-pharmaceuticals/12-mln-polio-vaccine-doses-delivered-gaza-ahead-sept-1-campaign-who-says-2024-08-30/",
        ),
        (
            "https://news.google.com/rss/articles/CBMi3AFBVV95cUxOX01TWDZZN2J5LWlmU3hudGZaRDh6a1dxUHMtalBEY1c0TlJSNlpieWxaUkxUU19MVTN3Y1BqaUZael83d1ctNXhaQUtPM0IyMFc4R3VydEtoMmFYMWpMU1Rtc3BjYmY4d3gxZHlMZG5NX0s1RmR2ZXI5YllvdzNSd2xkOFNCUTZTaEp3b0IxZEJZdVFLUDBNMC1wNGgwMGhjRG9HRFpRZU5BMFVIYjZCOWdWcHI1YzdoVHFWYnZSOEFwQ0NubGx3Rzd0SHN6OENKMXZUcHUxazA5WTIw?hl=en-US&gl=US&ceid=US%3Aen",
            "https://nltimes.nl/2024/08/25/disney-cruise-ship-sails-past-amsterdam-due-extinction-rebellion-blockade",
        ),
    ]
    encoded_urls = [pair[0] for pair in sample_urls]
    expected_decoded_urls = [pair[1] for pair in sample_urls]

    # Act
    articles_params = [
        gnews_utils.get_decoding_params(urlparse(url).path.split("/")[-1])
        for url in encoded_urls
    ]
    decoded_urls = gnews_utils.decode_urls(articles_params)

    # Assert
    assert len(decoded_urls) == 2
    for url in decoded_urls:
        assert url.startswith("http")
        assert "google.com" not in url
        assert url in expected_decoded_urls

    encoded_urls = [sample_article["url"]]
    expected_decoded_url = (
        "https://www.businessinsider.com/trump-campaign-2024-election-family-2023-10"
    )
    articles_params = [
        gnews_utils.get_decoding_params(urlparse(url).path.split("/")[-1])
        for url in encoded_urls
    ]
    decoded_urls = gnews_utils.decode_urls(articles_params)

    assert len(decoded_urls) == 1
    assert decoded_urls[0] == expected_decoded_url
    print(decoded_urls[0])


def test_get_full_article(sample_article):
    # Arrange
    gnews = GNews()

    # Act
    articles_params = [
        gnews_utils.get_decoding_params(
            urlparse(sample_article["url"]).path.split("/")[-1]
        )
    ]
    replaced_url = gnews_utils.decode_urls(articles_params)[0]
    print(f"{replaced_url=}")
    result = gnews.get_full_article(replaced_url)

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
