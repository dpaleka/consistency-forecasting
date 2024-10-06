from dotenv import load_dotenv
import argparse
import os
import asyncio
from datetime import datetime
from fq_from_news.news_processing_utils import (
    download_news_from_api,
    process_news,
)
from fq_from_news.fq_from_news_utils import (
    set_save_directories,
    generate_rough_forecasting_data,
    generate_final_forecasting_questions,
    verify_final_forecasting_questions,
)
from fq_from_news.date_utils import (
    parse_date,
    get_month_date_range,
    parse_datetime,
)

load_dotenv()

async def generate_forecasting_questions(articles_download_path, args):
    # Set the save directories for rough and final FQ generation
    set_save_directories(args.rough_fq_save_directory, args.final_fq_save_directory)

    # Generating the rough intermediate forecasting questions
    if args.gen_rough:
        await generate_rough_forecasting_data(
            articles_download_path,
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
            args.pose_date,
        )

    # Generating the final forecasting questions
    if args.gen_final:
        await generate_final_forecasting_questions(
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
            args.final_fq_gen_model_name,
            args.pose_date,
            args.fq_creation_date,
            args.be_lax_in_res_checking,
        )

    if args.verify_fqs:
        await verify_final_forecasting_questions(
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.final_fq_gen_model_name,
            args.final_fq_verification_model_name,
            args.news_source,
            args.be_lax_in_res_checking,
            args.verified_fq_save_directory,
        )


def main(args: argparse.Namespace) -> None:
    """
    Pipeline for generating forecasting question using News API downloaded articles.

    :args: Arguments supplied to the main function

    :returns: None
    """

    # If neither is set and not asked to only download news
    if (
        (not args.only_gen_rough)
        and (not args.only_gen_final)
        and (not args.only_verify_fq)
    ):
        args.gen_rough = args.gen_final = args.verify_fqs = True
    else:
        args.gen_rough = args.only_gen_rough
        args.gen_final = args.only_gen_final
        args.verify_fqs = args.only_verify_fq

    # Download the articles (skips if already downloaded)
    _ = download_news_from_api(
        args.start_date, args.end_date, args.num_pages, os.getenv("NEWS_API_KEY")
    )

    # Process the articles
    processed_news_articles_path = process_news(
        args.start_date,
        args.end_date,
        args.num_pages,
        args.news_new_threshold,
        args.processed_news_save_directory,
    )

    # If asked to only download news, return here
    if args.only_download_news:
        return

    # Set number of articles to be generated to a very large number if using all articles
    if args.num_articles == -1:
        args.num_articles = float("inf")

    # Check whether the user does NOT want to use different models for the two steps
    if args.rough_fq_gen_model_name == "":
        args.rough_fq_gen_model_name = args.model_name
    if args.final_fq_gen_model_name == "":
        args.final_fq_gen_model_name = args.model_name
    if args.final_fq_verification_model_name == "":
        args.final_fq_verification_model_name = args.model_name

    asyncio.run(generate_forecasting_questions(processed_news_articles_path, args))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # News source
    parser.add_argument(
        "--news-source",
        choices=["NewsAPI", "sentinel"],
        default="NewsAPI",
        help="Source of news data",
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model used for generating the FQ from the articles",
        default="gpt-4o-2024-05-13",
    )
    parser.add_argument(
        "--rough-fq-gen-model-name",
        type=str,
        help="""
        Overrides the value set by --model-name to use a separate model for generating 
        rough intermediate forecasting question data.
        """,
        default="",
    )
    parser.add_argument(
        "--final-fq-gen-model-name",
        type=str,
        help="""
        Overrides the value set by --model-name to use a separate model for generating 
        final forecasting questions.
        """,
        default="",
    )
    parser.add_argument(
        "--final-fq-verification-model-name",
        type=str,
        help="""
        Overrides the value set by --model-name to use a separate model for verifying 
        final forecasting questions.
        """,
        default="",
    )

    # Resolution checking
    parser.add_argument(
        "-lax",
        "--be-lax-in-res-checking",
        action="store_true",
        help="Should the Final Generator be lax while checking the resolution",
    )

    # Pose Date of the forecaster
    parser.add_argument(
        "--pose-date",
        type=parse_date,
        help="""
        Pose date for downloading news in YYYY-MM-DD format.

        The purpose of this date is to make the LLM add additional context about events that wouldn't have been known about by this date.
        """,
        default=datetime(2023, 10, 1),
    )

    # Created dates of the FQs
    parser.add_argument(
        "--fq-creation-date",
        type=parse_datetime,
        help="""
        Creation date for the FQs in YYYY-MM-DD-HH-MM-SS. This parameter is set to be the created_date parameter in the FQs.
        """,
        default=datetime(2024, 6, 30, 23, 59, 59),
    )

    # Arguments to permit exclusive actions
    exclusive_pipeline_group = parser.add_mutually_exclusive_group()
    exclusive_pipeline_group.add_argument(
        "--only-gen-rough",
        action="store_true",
        help="Set to True if only the intermediate rough forecasting questions should be generated and downloaded.",
        default=False,
    )
    exclusive_pipeline_group.add_argument(
        "--only-gen-final",
        action="store_true",
        help="""
        Set to True if the intermediate rough forecasting questions have already been downloaded 
        and you wish to create the final forecasting questions from them.  
        """,
        default=False,
    )
    exclusive_pipeline_group.add_argument(
        "--only-download-news",
        action="store_true",
        help="""
        Set to True to only download the news articles and undertake no further steps.  
        """,
        default=False,
    )
    exclusive_pipeline_group.add_argument(
        "--only-verify-fq",
        action="store_true",
        help="""
        Set to True to only verify the final forecasting questions using the common FQ verifier.
        """,
        default=False,
    )

    # Save directories
    parser.add_argument(
        "--rough-fq-save-directory",
        type=str,
        help="""
        Directory where to store the rough FQ data. If left empty, defaults to `data/news_feed_fq_generation/news_api/rough_forecasting_question_data`
        """,
        default="",
    )
    parser.add_argument(
        "--final-fq-save-directory",
        type=str,
        help="""
        Directory where to store the final unverified FQ data. If left empty, defaults to `data/news_feed_fq_generation/news_api/final_unverified`
        """,
        default="",
    )
    parser.add_argument(
        "--verified-fq-save-directory",
        type=str,
        help="""
        Directory where to store the verified FQ data. If left empty, defaults to `data/fq/synthetic/news_api_generated_fqs`
        """,
        default="",
    )
    parser.add_argument(
        "--processed-news-save-directory",
        type=str,
        help="""
        Directory where to store the processed news. If left empty, defaults to `data/news_feed_fq_generation/news_api/news_feed_data_dump`
        """,
        default="",
    )

    args, _ = parser.parse_known_args()

    # Conditionally add arguments based on the value of news_source
    if args.news_source == "NewsAPI":
        print("Using NewsAPI to download news")
        parser.add_argument(
            "--num-pages",
            type=int,
            help="""
            News API returns data in a paginated form. We set the number of articles downloaded per page to a 100 (maximum),
            By default, we only download the first page. 

            Set to the number of pages to be downloaded. Set to -1 to download all pages.
            """,
            default=1,
        )
        parser.add_argument(
            "--num-articles",
            type=int,
            help="""
            Set to the number of downloaded articles to be used to form the rough intermediate FQ data. 
            Set to -1 to use all articles.
            
            In case of generating only final FQs, set to number used to generate rough FQ data. 
            """,
            default=-1,
        )
        parser.add_argument(
            "--start-date",
            type=parse_date,
            help="Start date for downloading news in YYYY-MM-DD format. Do NOT use if using news month.",
        )
        parser.add_argument(
            "--end-date",
            type=parse_date,
            help="End date for downloading news in YYYY-MM-DD format. Do NOT use if using news month.",
        )
        parser.add_argument(
            "--news-month",
            type=int,
            choices=range(1, 13),
            help="Month for downloading news (1-12)",
        )
        parser.add_argument(
            "--news-year", type=int, help="Year for downloading news", default=2024
        )
        parser.add_argument(
            "--news-new-threshold",
            type=float,
            help="The threshold used to designate an article as a duplicate while processing",
            default=0.25,
        )
    else:
        raise NotImplementedError("Only NewsAPI support exists.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    assert (
        args.num_articles == -1 or args.num_articles > 0
    ), "Set a positive number or -1 for --num-articles!"

    # Validate the arguments to ensure either start_date and end_date are provided, or news_month and year
    if (args.start_date and args.end_date) and (args.news_month):
        raise RuntimeError(
            "You must provide either start and end dates or a news month and year, not both."
        )
    elif args.start_date and args.end_date:
        pass
    elif args.news_month and args.news_year:
        args.start_date, args.end_date = get_month_date_range(
            args.news_year, args.news_month
        )
    else:
        raise RuntimeError(
            "You must provide either (start and end dates) or (a news month and year)."
        )

    main(args)
