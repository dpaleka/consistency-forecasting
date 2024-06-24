import requests
import json
import datetime as dt
import argparse


def scrape_manifold_markets(
    api_url, start_date=None, end_date=None, max_pages=1, num_questions=500
):
    """
    Fetches market data from the Manifold Markets API and extracts relevant information.
    :param api_url: URL of the Manifold Markets API endpoint.
    :param start_date: Start date in 'yyyymmdd' format.
    :param end_date: End date in 'yyyymmdd' format.
    :param max_pages: Maximum number of pages to fetch from the API.
    :param num_questions: Number of questions to fetch.
    :return: A dictionary containing the extracted market names, their URLs, and other relevant data.
    """
    # Convert start_date and end_date to datetime objects for comparison
    start_date = dt.datetime.strptime(start_date, "%Y%m%d") if start_date else None
    end_date = dt.datetime.strptime(end_date, "%Y%m%d") if end_date else None

    headers = {"User-Agent": "Mozilla/5.0"}
    questions_info = []
    total_questions = num_questions  # Total number of questions you want to fetch
    page_size = 1000  # Number of questions per page. Adjust based on API's maximum allowed limit

    seen_ids = set()
    before_id = None
    for p in range(max_pages):
        if len(questions_info) >= total_questions:
            break

        params = {"limit": page_size, "before": before_id}
        # Fetch the content from the API URL
        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch the API data: {api_url}")
        # Parse the JSON content
        data = response.json()

        # print(p)
        # print(data)

        # Extract relevant data from each market
        for market in data:
            if len(questions_info) >= total_questions:
                break

            if market.get("outcomeType", "").lower() != "binary":
                continue

            resolution_date = None
            if market.get("closeTime"):
                timestamp = market["closeTime"] / 1000
                # Ensure the timestamp is within a reasonable range (e.g., after year 1970)
                if dt.datetime(9999, 12, 31).timestamp() > timestamp > 0:
                    resolution_date = dt.datetime.fromtimestamp(timestamp, dt.UTC)

            # Skip the market if its resolution date is not within the specified range
            if not resolution_date or (
                (start_date and resolution_date.replace(tzinfo=None) < start_date)
                or (end_date and resolution_date.replace(tzinfo=None) >= end_date)
            ):
                continue

            market_info = {
                "id": market["id"],
                "title": market["question"],
                "body": market.get("description", "N/A"),
                "question_type": market.get("outcomeType", "").lower(),
                "resolution_date": resolution_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "url": market.get("url", "").lower(),
                "data_source": "manifold",
                "metadata": {"topics": market.get("tags", [])},
                "resolution": market.get("isResolved", None),
            }

            market_info["metadata"][
                "api_url"
            ] = "https://api.manifold.markets/v0/slug/{}".format(
                market_info["url"].split("/")[-1]
            )
            market_info["metadata"]["market_prob"] = None

            if not market_info["resolution"]:
                market_info["resolution"] = None

            else:
                if market["resolution"].lower() == "yes":
                    market_info["resolution"] = True
                elif market["resolution"].lower() == "no":
                    market_info["resolution"] = False
                else:
                    market_info["resolution"] = None

            questions_info.append(market_info)

        before_id = market["id"]

    return questions_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch live questions from Manifold API."
    )
    parser.add_argument("-start", type=str, help="Start date in yyyymmdd format.")
    parser.add_argument("-end", type=str, help="End date in yyyymmdd format.")
    parser.add_argument(
        "-pages", type=str, help="Max pages of 1000 entries to go through.", default=10
    )
    parser.add_argument(
        "-num", type=int, help="Number of questions to fetch.", default=500
    )
    args = parser.parse_args()

    api_url = "https://api.manifold.markets/v0/markets"

    try:
        # Scrape the Manifold Markets
        data = scrape_manifold_markets(
            api_url, args.start, args.end, args.pages, args.num
        )

        print("total entries:", len(data))
        # Convert the data to JSON and print

        if args.start or args.end:
            with open(
                "manifold_{}_{}.json".format(str(args.start), str(args.end)), "w"
            ) as json_file:
                json.dump(data, json_file, indent=4)

        else:
            with open("manifold.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

    except Exception as e:
        print(f"Error: {e}")
