import subprocess
from datetime import datetime, timedelta

"""
Script to generate and execute commands for downloading news daily
between a given start date and end date, and define a function to generate file paths for saving
final forecasting questions.

Usage:
    python run_news_downloads.py

    NOTE: set the dates and the env variable in the code
"""


def daterange(start_date, end_date):
    """
    Generate date ranges in steps of one day.
    """
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(1)


def generate_commands(start_date, end_date):
    """
    Create the list of commands for each day.
    """
    commands = []
    for start in daterange(start_date, end_date):
        end = start  # For daily, start and end are the same
        command = f"python generate_fqs_from_news.py --start-date {start.strftime('%Y-%m-%d')} --end-date {end.strftime('%Y-%m-%d')} --only-download-news"
        commands.append(command)
    return commands


def main():
    # Define your start and end dates here
    start_date = datetime.strptime("2024-07-30", "%Y-%m-%d")
    end_date = datetime.strptime(
        "2024-08-03", "%Y-%m-%d"
    )  # next time set it to the next one
    subprocess.run(["conda", "activate", "spar"], shell=True)
    commands = generate_commands(start_date, end_date)
    for command in commands:
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
