import json
import os
from datetime import datetime
from uuid import uuid4
import pytz
from common.datatypes import ForecastingQuestion
from common.llm_utils import answer_sync, answer
from .fq_from_news_datatypes import (
    ForecastingQuestion_stripped_with_resolution,
    ForecastingQuestionGroundTruthResolution,
)
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

    initial_prompt = {
        "preface": """
        You are an expert in validating forecasting (prediction) questions that require nuanced understanding and cannot be answered by simple or trivial forecasting algorithms. Verify that each forecasting question's title, body, and resolution adhere to the established guidelines. Ensure questions are based on concrete events and assume the forecaster's current date is sometime in the previous year. Correct any discrepancies or ambiguities. The resolution date for the questions should be set as {month_name}, {year}.

        The forecaster will assume the current date is the `pose_date`, which is {pose_date}. Therefore, use concrete events to form your questions. If context or event names would not be apparent at the `pose_date`, provide sufficient context within the body to avoid revealing that the question was formed later. The simplest litmus test is to check whether you know of the event through solely your training data.
        """,
        "prompt": """
        You are tasked with following the validation process as described. 

        **Guidelines for Validating Forecasting Questions**

            1. **Definitive, Correct Answers**:
            - Questions must yield a clear YES or NO answer based on concrete, factual information from past news articles.
            - The resolution should align with this information, treating True as "Yes" and False as "No."
            - Ensure the resolution remains valid for all potential scenarios up to the specified resolution date (e.g., "by the end of July 2024").
            - Reject the question if the resolution is not correct in the context of the question's body and the news article.

            2. **Numeric Values**:
            - For questions involving numeric values, frame inquiries to determine if the value will cross or fall below a specified threshold.
            - Utilize rough thresholds to minimize numerical biases.

            3. **Objectivity and Clarity**:
            - Avoid ambiguous or subjective terms such as "significant", "major", or "substantial". 
            - Questions must be objective and unambiguous, steering clear of biases related to religion, culture, politics, or stereotypes.

            4. **Question Body**:
            - The body of the question should prompt a clear YES or NO resolution, avoiding overly specific details or easily inferred predictions.
            - Provide adequate context without leading to predictable answers.

            5. **Future Accuracy**:
            - Ensure that the question remains accurate for all foreseeable futures up to the specified resolution date.
            - Questions based on specific events are permissible, but they must not resolve between the forecaster's `pose_date` and the resolution date's month.

            6. **Avoid Overly Specific Questions**:
            - Questions should not depend on specific knowledge that could disadvantage certain models or participants. A question is overly specific if it references more than three distinct named entities from the source article.

            7. **Do Not Fabricate Information**:
            - Ensure that all questions are based solely on the information provided in the article. Avoid creating questions that stem from personal insights or interpretations beyond the content of the source.

            8. **Avoid Obvious Answers**:
            - Questions should not be so obvious that they can be easily guessed using common sense or simplistic reasoning.
            - Strive for a level of complexity that challenges the forecaster while remaining grounded in factual events.

                **Examples of obvious questions to avoid**:
                - "Will the sun rise tomorrow?"
                - "Will a country exist on Earth by August 2024?"
                - "Will a person named John Smith be born by August 2024?"
                - "Will a major earthquake occur in California by August 2024?"
                - "Will a new event happen named `xyz`?" (Rejected due to name specificity)

                **Examples of better questions**:
                - "Will a new country be formed by merging two existing countries by August 2024?"
                - "Will a major new international treaty be signed by at least 50 countries by August 2024?"
                - "Will a new type of renewable energy source provide more than 10% of a country's total electricity generation by August 2024?"
                - "Will a new vaccine for a previously untreatable disease be approved for public use by August 2024?"

            9. **No Reference to Articles**:
            - The title and body of the question must not reference any articles or indicate that the question was formed using news content.
            - Ensure there is no discernible knowledge of the `pose_date`.
            - Directly reject questions that fail this test without attempting modifications.

        **Validation Process:**

        1. Validation:
        - Validate the forecasting question and return its "Final Form." A question is valid if it follows the above guidelines.

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

        Think carefully & aptly and perform the above validation steps for the following forecasting question:
        ```JSON
        {source_rough_fq_data}
        ```
        """,
    }

    resolution_checker_prompt = {
        "preface": """
        You are an AI agent tasked with answering questions based solely on the content of a provided news article.
        
        Guidelines:
        - Respond only using information directly from the article.
        - Avoid adding any personal insights, interpretations, or external information.
        - Do not fabricate any details.
        """,
        "prompt": """
        Consider the following article:
            Title: {article_title}
            Description: {article_description}
            Content: {article_content}

        Consider this question - {question_title}
        Use the following information for disambiguating the above question:
            {question_body}

        You are tasked with answering the question using only factual information from the news article. Return
        1. `True`, if the answer to the question is Yes
        2. `False`, if the answer to the question is No
        """,
    }

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

    @classmethod
    def check_if_fq_was_rejected(cls, fq: ForecastingQuestion_stripped_with_resolution):
        return fq is None or fq.title.strip() == ""

    @classmethod
    def _processed_rough_fq_data(cls, rough_fq_date: dict) -> dict:
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

    @classmethod
    def _initial_prompt_and_preface_formation(
        cls, rough_fq_data: dict, end_date: datetime, pose_date: datetime
    ) -> tuple[str, str]:
        """
        Forms the initial final FQ generation forecasting prompt and preface from rough forecasting question data.

        Args:
            rough_fq_data (dict): Processed rough forecasting question data.
            end_date (datetime): The end date used to set context for the forecasting question.
            pose_date (datetime): The date assumed to be the knowledge cutoff for the forecaster.

        Returns:
            tuple[str, str]: A tuple containing the forecasting preface and prompt as strings.
        """
        forecasting_preface = cls.initial_prompt["preface"].format(
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            pose_date=pose_date.strftime("%B %d, %Y"),
        )
        forecasting_prompt = cls.initial_prompt["prompt"].format(
            source_rough_fq_data=rough_fq_data,
            example_fq_1=json.dumps(cls.example_fq_1, indent=4),
            example_fq_2=json.dumps(cls.example_fq_2, indent=4),
            example_fq_3=json.dumps(cls.example_fq_3, indent=4),
            example_rejected_fq=json.dumps(cls.example_rejected_fq, indent=4),
            rough_fq_data_desc=json.dumps(cls.rough_fq_data_desc, indent=4),
            final_fq_form=json.dumps(cls.final_fq_form, indent=4),
        )
        return forecasting_preface, forecasting_prompt

    @classmethod
    def _resolution_checker_prompt_and_preface_formation(
        cls,
        article_title,
        article_description,
        article_content,
        res_unchecked_fq_title,
        res_unchecked_fq_body,
    ) -> tuple[str, str]:
        """
        Forms the resolution checking forecasting prompt and preface from rough forecasting question data.
        """
        forecasting_preface = cls.resolution_checker_prompt["preface"]
        forecasting_prompt = cls.resolution_checker_prompt["prompt"].format(
            article_title=article_title,
            article_description=article_description,
            article_content=article_content,
            question_title=res_unchecked_fq_title,
            question_body=res_unchecked_fq_body,
        )
        return forecasting_preface, forecasting_prompt

    @classmethod
    def _form_final_fq_from_llm_return_val(
        cls,
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
        if cls.check_if_fq_was_rejected(generated_stripped_final_forecasting_question):
            return None

        cet_timezone = pytz.timezone("CET")
        scraped_date = datetime.now(cet_timezone).strftime("%Y-%m-%d %H:%M:%S")

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
                "scraped_date": scraped_date,
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
        ) = cls._initial_prompt_and_preface_formation(
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
    async def _rough_fq_to_resolution_unchecked_final_stripped_fq(
        cls,
        rough_fq_data: dict,
        model_name: str,
        end_date: datetime,
        pose_date: datetime,
    ) -> ForecastingQuestion_stripped_with_resolution:
        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._initial_prompt_and_preface_formation(
            cls._processed_rough_fq_data(rough_fq_data), end_date, pose_date
        )

        # Generate initial forcasting question (also attempts to verify resolution)
        stripped_resolution_unchecked_fq: ForecastingQuestion_stripped_with_resolution = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution,
        )

        return stripped_resolution_unchecked_fq

    @classmethod
    async def _res_unchecked_to_res_checked_final_stripped_fq(
        cls,
        rough_fq_data: dict,
        model_name: str,
        final_resolution_unchecked_forecasting_question: ForecastingQuestion_stripped_with_resolution,
    ) -> ForecastingQuestion_stripped_with_resolution:
        # If the FQ is already deemed invalid, return None
        if cls.check_if_fq_was_rejected(
            final_resolution_unchecked_forecasting_question
        ):
            return None

        # Form prompt and verify resolution
        processed_rough_fq_data = cls._processed_rough_fq_data(rough_fq_data)
        article_title, article_description, article_content = (
            processed_rough_fq_data["articleTitle"],
            processed_rough_fq_data["articleDescription"],
            processed_rough_fq_data["articleContent"],
        )

        res_unchecked_fq_title, res_unchecked_fq_body = (
            final_resolution_unchecked_forecasting_question.title,
            final_resolution_unchecked_forecasting_question.body,
        )

        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._resolution_checker_prompt_and_preface_formation(
            article_title,
            article_description,
            article_content,
            res_unchecked_fq_title,
            res_unchecked_fq_body,
        )

        # Generate fqs where resolution has been specifically checked
        generated_resolution: ForecastingQuestionGroundTruthResolution = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestionGroundTruthResolution,
        )

        if (
            generated_resolution.resolution
            != final_resolution_unchecked_forecasting_question.resolution
        ):
            return None

        return final_resolution_unchecked_forecasting_question

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

        # Generate the FQs post the initial final check
        final_resolution_unchecked_forecasting_question: ForecastingQuestion_stripped_with_resolution = await cls._rough_fq_to_resolution_unchecked_final_stripped_fq(
            rough_fq_data, model_name, end_date, pose_date
        )

        # Generate the resolution checked (not verified) forecasting questions
        final_resolution_checked_forecasting_question: ForecastingQuestion_stripped_with_resolution = await cls._res_unchecked_to_res_checked_final_stripped_fq(
            rough_fq_data, model_name, final_resolution_unchecked_forecasting_question
        )

        # Generate the forecasting question in the proper form
        final_fq: ForecastingQuestion = cls._form_final_fq_from_llm_return_val(
            rough_fq_data,
            final_resolution_checked_forecasting_question,
            end_date,
            pose_date,
        )

        return final_fq

    @classmethod
    def rough_fq_to_final_fq_download_path(
        cls,
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
            cls.news_api_final_fq_save_dir,
            news_save_file_name,
        )
