# Prediction Market Web Scraper

This project is designed to scrape prediction market websites for data and store this information in a local database. It supports multiple prediction market platforms and is configurable to add more as needed.

## Features

- Fetch market data from configurable prediction market websites.
- Parse the fetched data into a structured format.
- Store the parsed data into a SQLite database.

## Configuration

Before running the scraper, you need to configure the `config.json` file according to your requirements. This file includes the list of prediction market websites to scrape, database configuration, and other settings.

### Example `config.json`:

```json
{
  "prediction_market_websites": [
    {
      "name": "PredictIt",
      "base_url": "https://www.predictit.org",
      "markets_endpoint": "/api/marketdata/all/",
      "headers": {
        "User-Agent": "Mozilla/5.0"
      }
    },
    {
      "name": "Betfair",
      "base_url": "https://www.betfair.com",
      "markets_endpoint": "/exchange/plus/en/politics-betting-2378961",
      "headers": {
        "User-Agent": "Mozilla/5.0"
      }
    }
  ],
  "database_config": {
    "host": "localhost",
    "user": "your_database_user",
    "password": "your_database_password",
    "database": "prediction_markets"
  },
  "scrape_interval_seconds": 3600,
  "log_file_path": "logs/scraper_log.txt"
}
```

## Installation

To set up your environment to run this scraper, you need to install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Scraper

To start the scraper, simply run the `scraper.py` script:

```bash
python scraper.py
```

This will initiate the scraping process based on the configuration provided in `config.json`. The data fetched from the prediction market websites will be parsed and stored in the SQLite database specified in the configuration.

## Project Structure

- `config.json`: Configuration file for prediction market websites and database.
- `request_handler.py`: Handles HTTP requests to fetch market data.
- `parser.py`: Parses the raw market data into a structured format.
- `data_model.py`: Defines the data model for the parsed market data.
- `database.py`: Manages database connections and operations.
- `scraper.py`: Main script that orchestrates the scraping process.
- `requirements.txt`: Lists the Python package dependencies.
- `README.md`: This file, providing an overview of the project and instructions.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
