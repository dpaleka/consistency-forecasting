import json
import os
from datetime import datetime
from uuid import uuid4
from common.datatypes import ForecastingQuestion
from common.llm_utils import answer_sync, answer
from .fq_from_news_datatypes import ForecastingQuestion_stripped_with_resolution


class FinalForecastingQuestionGenerator:
    """
    Class with functionality to take the dicts formed by the RoughForecastingQuestionGenerator class
    and parse them to either accept, reject or improve them and then convert them into ForecastingQuestions
    """

    news_api_final_fq_save_dir = "./data/fq/synthetic/news_api_generated_fqs"
    # Create the save path directory
    os.makedirs(news_api_final_fq_save_dir, exist_ok=True)

    preface = """
    You are an expert in validating forecasting (prediction) questions that cannot be answered by a simple or trivial forecasting algorithm leveraging biases 
    and some other rules that will be given. The foreacsting questions have been made using News Articles.
    Assume that today's date is {current_date} in %Y-%m-%d format. The resolution dates of all the questions must be on or after this. 
    """

    prompt = """
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

    Here are the guidelines for a forecasting question to be "valid".
    1. The questions should definitively answer YES or NO based on factual information from recent news articles, confirming the outcome without room for speculation. Ensure the resolution aligns with the article's information, treating True as "Yes" and False as "No".

    2. Questions must accurately forecast binary outcomes using recent news articles. The resolution should be valid for all potential scenarios up to the resolution date. For questions involving numeric values where uncertainty exists (e.g., terms like "over" or "at least"), frame the question inversely. For instance, if the article states a value is above 47, valid questions could ask if the value will be above 40 (resolving to Yes) or below 45 (resolving to No).

    3. Questions must be objective and unambiguous, avoiding specificity that could allow simplistic algorithms to exploit biases. Ambiguous terms such as "significant," "major," "substantial," "many," or "few" should not be used without universally accepted values. Biases to avoid include those based on religion, cultural norms, political views, historical context, precise numerical thresholds, or stereotypes.

    4. Forecasting question bodies must prompt a resolution of either "Yes" or "No".

    5. Questions should not be overly specific, where a simple algorithm could guess the answer based on the information provided without requiring a nuanced understanding of current events. 
       Overly specific questions have fallacies including:
        a. They focus on niche, relatively unknown individuals or events that are not widely covered in mainstream news, making the outcome predictable to algorithms monitoring niche sources.
        b. They involve specific age-related outcomes which are personal and not widely reported unless the individual is highly publicized.
        c. They pertain to future events that are too far out or specific in nature, which are less likely to be covered extensively in current news, thus making the outcome easier to predict for algorithms.
       Examples of overly specific questions that are NOT valid include:
        a. Will an appeals court reject Garth Drabinsky's antitrust lawsuit against Actors' Equity by the end of this month?
        b. Will Pål Enger's death at age 57 be reported in major news outlets this month?
        c. Will Tarmo Peltokoski start his term as music director of the Hong Kong Philharmonic in the 2026-27 season?


    If a forecasting question follows the above guidelines, it is said to be validated.

    From this, you must do the following based on the guidelines given above: 
    1. Validate the forecasting question and return its "Final Form". 
    2. If the forecasting question is not valid, you must attempt to modify its data to make it valid and return the modified version's "Final Form"
    3. If the attempt to modify fails for invalid forecasting questions, you must return a rejected "Final Form" which looks like
        ```JSON
        {example_rejected_fq}
        ```
    
    Think carefully and aptly and do the above steps for the following forecasting question:
    ```JSON
    {source_rough_fq_data}
    ```
    """

    rough_fq_data_desc = {
        "articleTitle": "The title of the news article that was used to form the forecasting question",
        "articleDescription": "The description of the news article that was used to form the forecasting question",
        "articleContent": "The content of the news article that was used to form the forecasting question",
        "articleUrl": "The URL of the news article that had been used to form the forecasting question",
        "articlePulishedAt": "The date and time the news article was published such as 2024-07-05T01:10:56Z",
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
        "title": "Will a significant political figure publicly endorse a theory related to string theory's implications for the universe by 2030?",
        "body": "This question will resolve as Yes if, by December 31, 2030, a significant political figure publicly endorses a theory related to string theory's implications for the universe. A \"significant political figure\" is defined as a current or former head of state, government, or a member of the executive cabinet of a G20 nation. The endorsement must be explicit, relating directly to string theory's implications for the universe, and must be made in a public forum, such as an official speech, published article, or a verified social media post. The endorsement must be reported by at least two reputable news organizations (BBC, The Guardian, New York Times, Washington Post). In the event of ambiguous statements, the resolution will be based on the consensus interpretation by these reporting news organizations. This question resolves as YES upon confirmation of such an endorsement.",
        "resolution": True,
    }
    example_fq_2 = {
        "title": "Will a First Crystal Tier Market be made in 2024?",
        "body": 'Resolves YES immediately when someone creates a Crystal tier market in 2024 (which currently costs 1 million Mana to make).\n\nResolves NO if no such market is created before January 1, 2025 (UTC time) or Manifold entirely scraps the tier system for creating questions (minor modifications don\'t alter the outcome, see below).\n\nImportant edge cases:\n\nFor the purposes of this market if Manifold alters the tier system prices, any questions created with a tier that has a creation cost of between 500k Mana and 2M Mana, inclusive, will be considered equivalent to the "Crystal tier" market.\n\nAny changes to the tier name will not be considered consequential (only the creation cost).',
        "resolution": False,
    }

    def _prompt_and_preface_formation(
        rough_fq_data: dict, start_date: datetime
    ) -> tuple[str, str]:
        forecasting_preface = FinalForecastingQuestionGenerator.preface.format(
            current_date=start_date.strftime("%Y-%m-%d")
        )
        forecasting_prompt = FinalForecastingQuestionGenerator.prompt.format(
            source_rough_fq_data=rough_fq_data,
            example_fq_1=json.dumps(
                FinalForecastingQuestionGenerator.example_fq_1, indent=4
            ),
            example_fq_2=json.dumps(
                FinalForecastingQuestionGenerator.example_fq_2, indent=4
            ),
            example_rejected_fq=json.dumps(
                FinalForecastingQuestionGenerator.example_rejected_fq, indent=4
            ),
            rough_fq_data_desc=json.dumps(
                FinalForecastingQuestionGenerator.rough_fq_data_desc, indent=4
            ),
            final_fq_form=json.dumps(
                FinalForecastingQuestionGenerator.final_fq_form, indent=4
            ),
        )
        return forecasting_preface, forecasting_prompt

    def _form_final_fq_from_llm_return_val(
        rough_fq_data: dict,
        generated_stripped_final_forecasting_question: ForecastingQuestion,
    ) -> ForecastingQuestion:
        if generated_stripped_final_forecasting_question.title.strip() == "":
            return None

        return ForecastingQuestion(
            id=uuid4(),
            title=generated_stripped_final_forecasting_question.title,
            body=generated_stripped_final_forecasting_question.body,
            resolution=generated_stripped_final_forecasting_question.resolution,
            question_type="binary",
            data_source="synthetic",
            url=rough_fq_data["articleUrl"],
            resolution_date=datetime.strptime(
                rough_fq_data["articlePulishedAt"], "%Y-%m-%dT%H:%M:%SZ"
            ),
            metadata={},
        )

    @classmethod
    def rough_fq_to_final_fq_sync(
        cls, rough_fq_data: dict, model_name: str, start_date: datetime
    ) -> ForecastingQuestion:
        """
        Classmethod to create the final ForecastingQuestion from the rough forecasting question data
        in a sync manner.

        :rough_fq_data: The rough intermediate forecasting question data
        :model_name: The model being used to create the rough forecasting question
        :start_date: Used to set context of the current date for the model

        :returns: validated (and possibly modified) ForecastingQuestion, or None
        """
        (
            forecasting_preface,
            forecasting_prompt,
        ) = FinalForecastingQuestionGenerator._prompt_and_preface_formation(
            rough_fq_data, start_date
        )

        generated_stripped_final_forecasting_question = answer_sync(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution,
        )

        return FinalForecastingQuestionGenerator._form_final_fq_from_llm_return_val(
            rough_fq_data, generated_stripped_final_forecasting_question
        )

    @classmethod
    async def rough_fq_to_final_fq(
        cls, rough_fq_data: dict, model_name: str, start_date: datetime
    ) -> ForecastingQuestion:
        """
        Classmethod to create the final ForecastingQuestion from the rough forecasting question data
        in an async manner.

        :rough_fq_data: The rough intermediate forecasting question data
        :model_name: The model being used to create the rough forecasting question
        :start_date: Used to set context of the current date for the model

        :returns: validated (and possibly modified) ForecastingQuestion, or None
        """
        (
            forecasting_preface,
            forecasting_prompt,
        ) = FinalForecastingQuestionGenerator._prompt_and_preface_formation(
            rough_fq_data, start_date
        )

        generated_stripped_final_forecasting_question = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution,
        )

        return FinalForecastingQuestionGenerator._form_final_fq_from_llm_return_val(
            rough_fq_data, generated_stripped_final_forecasting_question
        )

    def rough_fq_to_final_fq_download_path(
        start_date: datetime,
        end_date: datetime,
        num_pages: int,
        num_articles: int,
        model_name: str,
    ) -> str:
        """
        File path to save the final forecasting questions

        :start_date: Start date for downloading news
        :end_date: End date for downloading news
        :num_pages: Number of pages of news that were downloaded
        :num_articles: Number of articles in use
        :model_name: The model being used to create the final forecasting questions

        :returns: file path for saving
        """
        if num_pages == -1:
            num_pages = "all"
        if num_articles == -1 or num_articles == float("inf"):
            num_articles = "all"

        news_save_file_name = f"final_fq_using_{model_name}_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}_num_articles_{num_articles}.jsonl"

        return os.path.join(
            FinalForecastingQuestionGenerator.news_api_final_fq_save_dir,
            news_save_file_name,
        )
