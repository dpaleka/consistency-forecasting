#!/bin/bash

# Activate the conda environment
conda activate spar

# Run the python scripts with the specified date ranges
python generate_fqs_from_news.py --start-date 2024-06-22 --end-date 2024-06-24 --only-download-news
python generate_fqs_from_news.py --start-date 2024-06-25 --end-date 2024-06-27 --only-download-news
python generate_fqs_from_news.py --start-date 2024-06-28 --end-date 2024-06-30 --only-download-news
python generate_fqs_from_news.py --start-date 2024-07-01 --end-date 2024-07-03 --only-download-news
python generate_fqs_from_news.py --start-date 2024-07-04 --end-date 2024-07-06 --only-download-news
python generate_fqs_from_news.py --start-date 2024-07-07 --end-date 2024-07-09 --only-download-news
python generate_fqs_from_news.py --start-date 2024-07-10 --end-date 2024-07-12 --only-download-news
python generate_fqs_from_news.py --start-date 2024-07-13 --end-date 2024-07-15 --only-download-news
python generate_fqs_from_news.py --start-date 2024-07-16 --end-date 2024-07-18 --only-download-news
