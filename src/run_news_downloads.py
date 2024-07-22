import subprocess
from datetime import datetime, timedelta

"""
Script to generate and execute commands for downloading news in sections of consecutive three days
between a given start date and end date, and define a function to generate file paths for saving
final forecasting questions.

Usage:
    python run_news_downloads.py

    NOTE: set the dates and the env variable in the code
"""


def daterange(start_date, end_date):
    """
    Generate date ranges in steps of three days.
    """
    for n in range(0, (end_date - start_date).days + 1, 3):
        yield start_date + timedelta(n)


def generate_commands(start_date, end_date):
    """
    Create the list of commands for each 3-day section.
    """
    commands = []
    for start in daterange(start_date, end_date):
        end = start + timedelta(2)
        if end > end_date:
            end = end_date
        command = f"python generate_fqs_from_news.py --start-date {start.strftime('%Y-%m-%d')} --end-date {end.strftime('%Y-%m-%d')} --only-download-news"
        commands.append(command)
    return commands


def rough_fq_to_final_fq_download_path(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    model_name: str,
) -> str:
    """
    Generate the file path to save the final forecasting questions.

    :start_date: Start date for downloading news
    :end_date: End date for downloading news
    :num_pages: Number of pages of news that were downloaded
    :num_articles: Number of articles in use
    :model_name: The model being used to create the final forecasting questions

    :returns: file path for saving
    """
    return f"fqs_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{num_pages}pages_{num_articles}articles_{model_name}.json"


def main():
    # Define your start and end dates here
    start_date = datetime.strptime("2024-06-22", "%Y-%m-%d")
    end_date = datetime.strptime("2024-07-18", "%Y-%m-%d")

    # Activate conda environment
    subprocess.run(["conda", "activate", "spar"], shell=True)

    # Generate and execute commands
    commands = generate_commands(start_date, end_date)
    for command in commands:
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
