import json
import os
from datetime import datetime
from uuid import uuid4
from common.datatypes import ForecastingQuestion
from common.llm_utils import answer_sync, answer
from .fq_from_news_datatypes import ForecastingQuestion_stripped_with_resolution
from .date_utils import last_datetime_of_month


class NewsApiFinalForecastingQuestionGenerator:
    """
    Class with functionality to take the dicts formed by the RoughForecastingQuestionGenerator class
    and parse them to either accept, reject or improve them and then convert them into ForecastingQuestions
    """

    news_api_final_fq_save_dir = (
        "./data/news_feed_fq_generation/news_api/final_unverified_forecasting_questions"
    )
    # Create the save path directory
    os.makedirs(news_api_final_fq_save_dir, exist_ok=True)

    preface = """
    You are an expert in validating forecasting (prediction) questions that require nuanced understanding and cannot be answered by simple or trivial forecasting algorithms. Verify that each forecasting question's title, body, and resolution adhere to the established guidelines. Ensure questions are based on concrete events and assume the forecaster's current date is sometime in the previous year. Correct any discrepancies or ambiguities. The resolution date for the questions should be set as {month_name}, {year}.

    The forecaster will assume the current date is the `pose_date`, which is {pose_date}. Therefore, use concrete events to form your questions. If context or event names would not be apparent at the `pose_date`, provide sufficient context within the body to avoid revealing that the question was formed later. The simplest litmus test is to check whether you know of the event through solely your training data.
    """

    prompt = """
    Guidelines for Validating Forecasting Questions

    1. Definitive Answers:
    - Questions must yield a clear YES or NO answer based on concrete, factual information from past news articles.
    - The resolution should align with this information, treating True as "Yes" and False as "No."
    - Ensure the resolution remains valid for all potential scenarios up to the specified resolution date (e.g., "by the end of July 2024").

    2. Numeric Values:
    - For questions involving numeric values, frame the inquiry to determine whether the value will cross or fall below a specified threshold.
    - Utilize rough thresholds to minimize numerical biases.

    3. Objectivity and Clarity:
    - Avoid ambiguous or subjective terms such as "significant," "major," or "substantial," unless universally accepted values are provided.
    - Questions must be objective and unambiguous, steering clear of biases related to religion, culture, politics, or stereotypes.

    4. Question Body:
    - The body of the question should prompt a clear YES or NO resolution, avoiding overly specific details or predictions that could be easily inferred by algorithms.
    - Provide adequate context without leading to predictable answers.

    5. Future Accuracy:
    - Ensure that the question remains accurate for all foreseeable futures up to the specified resolution date.
    - Questions based on specific events are permissible, but they must not resolve between the forecaster's current date (a date in the previous year) and the resolution date's month.

    6. Avoid Obvious Answers:
    - Questions should not be so obvious that they can be easily guessed using common sense or simple heuristics.
    - Strive for a level of complexity that challenges the forecaster while remaining grounded in factual events.

    7. No Reference to Articles:
    - The title and body of the question must not reference any articles or indicate that the question was formed using news content.
    - Ensure there is no discernible knowledge of the `pose_date`.
    - Directly reject questions that fail this test without attempting modifications.

    Validation Process:

    1. Validation:
    - Validate the forecasting question and return its "Final Form."

    2. Modification:
    - If the question is invalid, attempt to modify it to meet the guidelines and return the modified version's "Final Form."

    3. Rejection:
    - If modification fails, return a rejected "Final Form." The rejected form is: 
        ```JSON
        {example_rejected_fq}
        ```

    Data will be given to you in the following form:
    ```JSON
    {rough_fq_data_desc}
    ```

    You must return the "Final Form" of the forecasting question. It is defined as:
    ```JSON
    {final_fq_form}
    ```

    Here are a few examples of forecasting questions in their "Final Form":
    ```JSON
    {example_fq_1}
    ```
    Example 2:
    ```JSON
    {example_fq_2}
    ```
    ```JSON
    {example_fq_3}
    ```

    Think carefully & aptly and perform the aforementioned steps for the following forecasting question:
    ```JSON
    {source_rough_fq_data}
    ```
    """

    rough_fq_data_desc = {
        "articleTitle": "The title of the news article that was used to form the forecasting question",
        "articleDescription": "The description of the news article that was used to form the forecasting question",
        "articleContent": "The content of the news article that was used to form the forecasting question",
        "fqTitle": "The title of the forecasting question",
        "fqBody": "The Body of the forecasting question",
        "fqResolution": "A boolean representing the resolution of the forecasting question. True for 'Yes' and False for 'No'",
    }

    final_fq_form = {
        "title": "The title of the forecasting question",
        "body": "The body of the forecasting question",
        "resolution": "The resolution of the forecasting question",
    }

    example_rejected_fq = {
        "title": "",
        "body": "<Reason for rejection>",
        "resolution": False,
    }

    example_fq_1 = {
        "title": "Will a significant political figure publicly endorse a theory related to string theory's implications for the universe by July 2024?",
        "body": "This question will resolve as YES if, by July 31, 2024, a significant political figure publicly endorses a theory related to string theory's implications for the universe. A \"significant political figure\" is defined as a current or former head of state, government, or a member of the executive cabinet of a G20 nation. The endorsement must be explicit, relating directly to string theory's implications for the universe, and must be made in a public forum, such as an official speech, published article, or a verified social media post. The endorsement must be reported by at least two reputable news organizations (BBC, The Guardian, New York Times, Washington Post). In the event of ambiguous statements, the resolution will be based on the consensus interpretation by these reporting news organizations. This question resolves as YES upon confirmation of such an endorsement.",
        "resolution": True,
    }

    example_fq_2 = {
        "title": "Will a First Crystal Tier Market be made by August 2024?",
        "body": 'Resolves YES immediately when someone creates a Crystal tier market in 2024 (which currently costs 1 million Mana to make).\n\nResolves NO if no such market is created before September 1, 2024 (UTC time) or Manifold entirely scraps the tier system for creating questions (minor modifications don\'t alter the outcome, see below).\n\nImportant edge cases:\n\nFor the purposes of this market if Manifold alters the tier system prices, any questions created with a tier that has a creation cost of between 500k Mana and 2M Mana, inclusive, will be considered equivalent to the "Crystal tier" market.\n\nAny changes to the tier name will not be considered consequential (only the creation cost).',
        "resolution": False,
    }

    example_fq_3 = {
        "title": "Will the TIME 100 Most Influential Companies of 2024 list actually come out in May 2024 as promised?",
        "body": 'This page has just said "2024 HONOREES ANNOUNCED IN MAY" for ages now. I\'ve been checking every day for my market, @/Joshua/what-will-be-time-magazines-100-mos-1ccb89e7e3a1 \n\nThe FAQ says:\n\n[image]This market closes at 11:59 PM PT on Friday, May 31, 2024. If the list is published before market close, resolves YES. If not, resolves NO.',
        "resolution": True,
    }

    def _processed_rough_fq_data(rough_fq_date: dict) -> dict:
        """
        Processes rough forecasting question data by removing unwanted fields.

        Args:
            rough_fq_date (dict): Dictionary containing the rough forecasting question data.

        Returns:
            dict: Processed dictionary with specific keys removed.
        """
        return {
            key: value
            for key, value in rough_fq_date.items()
            if key not in ["articleUrl", "articlePublishedAt"]
        }

    def _prompt_and_preface_formation(
        rough_fq_data: dict, end_date: datetime, pose_date: datetime
    ) -> tuple[str, str]:
        """
        Forms the forecasting prompt and preface from rough forecasting question data.

        Args:
            rough_fq_data (dict): Processed rough forecasting question data.
            end_date (datetime): The end date used to set context for the forecasting question.
            pose_date (datetime): The date assumed to be the knowledge cutoff for the forecaster.

        Returns:
            tuple[str, str]: A tuple containing the forecasting preface and prompt as strings.
        """
        forecasting_preface = NewsApiFinalForecastingQuestionGenerator.preface.format(
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            pose_date=pose_date.strftime("%B %d, %Y"),
        )
        forecasting_prompt = NewsApiFinalForecastingQuestionGenerator.prompt.format(
            source_rough_fq_data=rough_fq_data,
            example_fq_1=json.dumps(
                NewsApiFinalForecastingQuestionGenerator.example_fq_1, indent=4
            ),
            example_fq_2=json.dumps(
                NewsApiFinalForecastingQuestionGenerator.example_fq_2, indent=4
            ),
            example_fq_3=json.dumps(
                NewsApiFinalForecastingQuestionGenerator.example_fq_3, indent=4
            ),
            example_rejected_fq=json.dumps(
                NewsApiFinalForecastingQuestionGenerator.example_rejected_fq, indent=4
            ),
            rough_fq_data_desc=json.dumps(
                NewsApiFinalForecastingQuestionGenerator.rough_fq_data_desc, indent=4
            ),
            final_fq_form=json.dumps(
                NewsApiFinalForecastingQuestionGenerator.final_fq_form, indent=4
            ),
        )
        return forecasting_preface, forecasting_prompt

    def _form_final_fq_from_llm_return_val(
        rough_fq_data: dict,
        generated_stripped_final_forecasting_question: ForecastingQuestion_stripped_with_resolution,
        end_date: datetime,
        pose_date: datetime,
    ) -> ForecastingQuestion:
        """
        Forms the final ForecastingQuestion from the LLM-generated stripped forecasting question.

        Args:
            rough_fq_data (dict): Processed rough forecasting question data.
            generated_stripped_final_forecasting_question (ForecastingQuestion_stripped_with_resolution): The LLM-generated stripped forecasting question.
            end_date (datetime): The end date used for the forecasting question resolution.
            pose_date (datetime): The assumed knowledge cutoff date for the forecaster.

        Returns:
            ForecastingQuestion: Validated and possibly modified ForecastingQuestion, or None if the title is empty.
        """
        if generated_stripped_final_forecasting_question.title.strip() == "":
            return None

        return ForecastingQuestion(
            id=uuid4(),
            title=generated_stripped_final_forecasting_question.title,
            body=generated_stripped_final_forecasting_question.body,
            resolution=generated_stripped_final_forecasting_question.resolution,
            question_type="binary",
            data_source="synthetic",
            url=None,
            resolution_date=last_datetime_of_month(end_date),
            metadata={
                "article_information": {
                    "article_url": rough_fq_data["articleUrl"],
                    "article_date": datetime.strptime(
                        rough_fq_data["articlePulishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                    ).strftime("%Y-%m-%d %H:%M:%S"),
                    "article_description": rough_fq_data["articleDescription"],
                    "article_title": rough_fq_data["articleTitle"],
                    "article_content": rough_fq_data["articleContent"],
                },
                "pose_date": pose_date.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    @classmethod
    def rough_fq_to_final_fq_sync(
        cls,
        rough_fq_data: dict,
        model_name: str,
        end_date: datetime,
        pose_date: datetime,
    ) -> ForecastingQuestion:
        """
        Class method to create the final ForecastingQuestion from rough forecasting question data synchronously.

        Args:
            rough_fq_data (dict): The rough intermediate forecasting question data.
            model_name (str): The model being used to create the rough forecasting question.
            end_date (datetime): Used to set context of the current date for the model.
            pose_date (datetime): The date assumed to be the knowledge cutoff for the forecaster.

        Returns:
            ForecastingQuestion: Validated and possibly modified ForecastingQuestion, or None if the title is empty.
        """
        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._prompt_and_preface_formation(
            cls._processed_rough_fq_data(rough_fq_data), end_date, pose_date
        )

        generated_stripped_final_forecasting_question = answer_sync(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution,
        )

        return cls._form_final_fq_from_llm_return_val(
            rough_fq_data,
            generated_stripped_final_forecasting_question,
            end_date,
            pose_date,
        )

    @classmethod
    async def rough_fq_to_final_fq(
        cls,
        rough_fq_data: dict,
        model_name: str,
        end_date: datetime,
        pose_date: datetime,
    ) -> ForecastingQuestion:
        """
        Class method to create the final ForecastingQuestion from rough forecasting question data asynchronously.

        Args:
            rough_fq_data (dict): The rough intermediate forecasting question data.
            model_name (str): The model being used to create the rough forecasting question.
            end_date (datetime): Used to set context of the current date for the model.
            pose_date (datetime): The date assumed to be the knowledge cutoff for the forecaster.

        Returns:
            ForecastingQuestion: Validated and possibly modified ForecastingQuestion, or None if the title is empty.
        """
        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._prompt_and_preface_formation(
            cls._processed_rough_fq_data(rough_fq_data), end_date, pose_date
        )

        generated_stripped_final_forecasting_question = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution,
        )

        return cls._form_final_fq_from_llm_return_val(
            rough_fq_data,
            generated_stripped_final_forecasting_question,
            end_date,
            pose_date,
        )

    def rough_fq_to_final_fq_download_path(
        start_date: datetime,
        end_date: datetime,
        num_pages: int,
        num_articles: int,
        model_name: str,
    ) -> str:
        """
        File path to save the final forecasting questions.

        Args:
            start_date (datetime): Start date for downloading news.
            end_date (datetime): End date for downloading news.
            num_pages (int): Number of pages of news that were downloaded.
            num_articles (int): Number of articles in use.
            model_name (str): The model being used to create the final forecasting questions.

        Returns:
            str: File path for saving the final forecasting questions.
        """
        if num_pages == -1:
            num_pages = "all"
        if num_articles == -1 or num_articles == float("inf"):
            num_articles = "all"

        model_name = model_name.replace("/", "__").replace("\\", "__")
        news_save_file_name = f"final_fq_using_{model_name}_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}_num_articles_{num_articles}.jsonl"

        return os.path.join(
            NewsApiFinalForecastingQuestionGenerator.news_api_final_fq_save_dir,
            news_save_file_name,
        )
