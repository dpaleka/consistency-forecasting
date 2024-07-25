from dotenv import load_dotenv
import argparse
import os
import asyncio
from fq_from_news.download_from_news_api import parse_date, download_news_from_api
from fq_from_news.fq_from_news_utils import (
    generate_rough_forecasting_data,
    generate_final_forecasting_question,
    generate_rough_forecasting_data_sync,
    generate_final_forecasting_question_sync,
)


load_dotenv()


def generate_forecasting_questions_from_news_sync(articles_download_path, args):
    # Generating the rough intermediate forecasting questions
    if args.only_gen_rough:
        generate_rough_forecasting_data_sync(
            articles_download_path,
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
        )

    # Generating the final forecasting questions
    if args.only_gen_final:
        generate_final_forecasting_question_sync(
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
            args.final_fq_gen_model_name,
        )


async def generate_forecasting_questions(articles_download_path, args):
    # Generating the rough intermediate forecasting questions
    if args.only_gen_rough:
        await generate_rough_forecasting_data(
            articles_download_path,
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
        )

    # Generating the final forecasting questions
    if args.only_gen_final:
        await generate_final_forecasting_question(
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
            args.final_fq_gen_model_name,
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
        and (not args.only_download_news)
    ):
        args.only_gen_rough = args.only_gen_final = True

    # Download the articles (skips if already downloaded)
    articles_download_path = download_news_from_api(
        args.start_date, args.end_date, args.num_pages, os.getenv("NEWS_API_KEY")
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

    if args.sync:
        generate_forecasting_questions_from_news_sync(articles_download_path, args)
    else:
        asyncio.run(generate_forecasting_questions(articles_download_path, args))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # News source
    parser.add_argument(
        "--news-source",
        choices=["NewsAPI", "sentinel"],
        default="NewsAPI",
        help="Source of news data",
    )

    # Placeholder for conditional arguments
    conditional_args = argparse.Namespace()
    conditional_args.NewsAPI = []
    conditional_args.sentinel = []

    # NewsAPI arguments
    conditional_args.NewsAPI.append(
        parser.add_argument(
            "--num-pages",
            type=int,
            help="""
            Use with NewsAPI.

            News API returns data in a paginated form. We set the number of articles downloaded per page to a 100 (maximum),
            By default, we only download the first page. 

            Set to the number of pages to be downloaded. Set to -1 to download all pages.
            """,
            default=1,
        )
    )
    conditional_args.NewsAPI.append(
        parser.add_argument(
            "--num-articles",
            type=int,
            help="""
            Use with NewsAPI.

            Set to the number of downloaded articles to be used to form the rough intermediate FQ data. 
            Set to -1 to use all articles.
            
            In case of generating only final FQs, set to number used to generate rough FQ data. 
            """,
            default=-1,
        )
    )
    conditional_args.NewsAPI.append(
        parser.add_argument(
            "--start-date",
            type=parse_date,
            help="""
            Use with NewsAPI.

            Start date for downloading news in YYYY-MM-DD format.
            """,
            required=True,
        )
    )
    conditional_args.NewsAPI.append(
        parser.add_argument(
            "--end-date",
            type=parse_date,
            help="""
            Use with NewsAPI.

            End date for downloading news in YYYY-MM-DD format.
            """,
            required=True,
        )
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

    # Sync enabling argument
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Set to True if FQs should be generated WITHOUT leveraging async (parallel) behaviour.",
        default=False,
    )

    # Arguments to permit exclusive actions
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only-gen-rough",
        action="store_true",
        help="Set to True if only the intermediate rough forecasting questions should be generated and downloaded.",
        default=False,
    )
    group.add_argument(
        "--only-gen-final",
        action="store_true",
        help="""
        Set to True if the intermediate rough forecasting questions have already been downloaded 
        and you wish to create the final forecasting questions from them.  
        """,
        default=False,
    )
    group.add_argument(
        "--only-download-news",
        action="store_true",
        help="""
        Set to True to only download the news articles and undertake no further steps.  
        """,
        default=False,
    )

    args = parser.parse_args()

    # Conditionally add arguments based on news_source
    if args.news_source == "NewsAPI":
        print("Using NewsAPI to download news")
        for arg in conditional_args.NewsAPI:
            parser.add_argument(arg)
    else:
        raise NotImplementedError("Sentinel scraping has not been implemented yet")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    assert (
        args.num_articles == -1 or args.num_articles > 0
    ), "Set a positive number or -1 for --num-articles!"

    main(args)
