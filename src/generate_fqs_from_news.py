from dotenv import load_dotenv
import argparse
import os
import asyncio
from fq_from_news.download_from_news_api import parse_date, download_news_from_api
from fq_from_news.fq_from_news_utils import (
    generate_rough_forecasting_data,
    generate_final_forecasting_questions,
    generate_rough_forecasting_data_sync,
    generate_final_forecasting_questions_sync,
    verify_final_forecasting_questions,
)


load_dotenv()


def generate_forecasting_questions_from_news_sync(articles_download_path, args):
    # Generating the rough intermediate forecasting questions
    if args.gen_rough:
        generate_rough_forecasting_data_sync(
            articles_download_path,
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
        )

    # Generating the final forecasting questions
    if args.gen_final:
        generate_final_forecasting_questions_sync(
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
            args.final_fq_gen_model_name,
        )

    if args.verify_fqs:
        raise NotImplementedError(
            "Cannot verify final forecasting questions in a sync manner!"
        )


async def generate_forecasting_questions(articles_download_path, args):
    # Generating the rough intermediate forecasting questions
    if args.gen_rough:
        await generate_rough_forecasting_data(
            articles_download_path,
            args.start_date,
            args.end_date,
            args.num_pages,
            args.num_articles,
            args.rough_fq_gen_model_name,
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
    if args.final_fq_verification_model_name == "":
        args.final_fq_verification_model_name = args.model_name

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

    # Sync enabling argument (cannot be used with verify)
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Set to True if FQs should be generated WITHOUT leveraging async (parallel) behaviour.",
        default=False,
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
            help="Start date for downloading news in YYYY-MM-DD format",
            required=True,
        )
        parser.add_argument(
            "--end-date",
            type=parse_date,
            help="End date for downloading news in YYYY-MM-DD format",
            required=True,
        )
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
