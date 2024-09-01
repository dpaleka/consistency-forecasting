import json
import os
from datetime import datetime
from common.llm_utils import answer_sync, answer
from common.datatypes import ValidationResult
from .fq_from_news_datatypes import ForecastingQuestion_stripped_with_resolution_list
from .date_utils import format_news_range_date


class NewsApiRoughForecastingQuestionGenerator:
    """
    Class with functionality to generate "rough" forecasting questions which may required to be iterated over later
    to prune out questions that do not guidelines for forming FQs such as the Navalny Problem.
    """

    news_api_rough_fq_save_dir = (
        "./data/news_feed_fq_generation/news_api/rough_forecasting_question_data"
    )
    # Create the save path directory
    os.makedirs(news_api_rough_fq_save_dir, exist_ok=True)

    news_validation_prompt = {
        "preface": """
        You are an AI agent tasked with evaluating news articles to determine if they are suitable for generating forecasting (prediction) questions that can be answered with a definitive YES or NO. Assess each article against the following criteria to ensure clarity, relevance, and factual accuracy:

        1. **Clarity of Content**: Does the article present information in a generally clear and straightforward manner? While some ambiguity is acceptable, reject articles that are overly convoluted or difficult to understand.

        2. **Definitive Events**: Does the article focus on concrete events that have occurred or are planned? Articles that reference past events not within the LLM's knowledge should still be evaluated based on the clarity and context they provide regarding those events.

        3. **Contextual Relevance**: Does the article provide sufficient context for the events discussed? Articles may have some gaps in background information, but they should still allow for a reasonable understanding of the events, even if they refer to past occurrences.

        4. **Specificity of Information**: Is the information specific enough to allow for the formulation of precise forecasting questions? While some generality is acceptable, reject articles that are too vague to support clear predictions.

        5. **Binary Resolution Potential**: Does the article imply a resolution that can be reasonably confirmed as TRUE (YES) or FALSE (NO)? Articles may contain some subjective elements, but they should still lead to a binary resolution.

        6. **Avoidance of Sensitive Topics**: Does the article avoid sensitive subjects like religion, politics, gender, or race? Reject articles that may introduce significant bias.

        7. **Completeness of Information**: Does the article contain enough detail to create multiple high-quality forecasting questions? It’s acceptable for articles to be somewhat brief, as long as they provide sufficient information for question generation.

        8. **Numerical Clarity**: If applicable, does the article present clear thresholds or metrics for numerical data? Some ambiguity in numerical references is acceptable, but they should still be understandable.

        9. **Sufficiency for Definitive Resolution**: The article should provide enough information to formulate forecasting questions that can yield resolutions remaining definitive from the current date until the specified resolution date in {month_name}, {year}. Ensure that the content supports actionable predictions based on concrete events, assuming the current date is {pose_date}.

        10. **Truncated Information**: Truncated information is NOT a cause for rejection. As long as the article can be used to form a prediction question, accept it, even if it references past events not covered by the LLM's knowledge.

        An article that meets most of these criteria is considered "complete" and suitable for generating forecasting questions, even if it contains some minor ambiguities or references past events that the you may not fully know.
        """,
        "prompt": """
        Please evaluate the following news article based on the established criteria for completeness. 
        {source_article}

        Based on your evaluation, determine if the article is "complete" and suitable for generating forecasting questions. Provide a brief justification for your assessment.
        """,
    }

    rough_fq_generation_prompt = {
        "preface": """
        You are tasked with generating forecasting questions that can be answered with a definitive YES or NO based on the provided news articles. Ensure each question is clear, unambiguous, and free from sensitive topics like religion, politics, or gender. Avoid subjective terms like "significant."

        Questions must have a resolution that remains definitive from the current date until {month_name}, {year}. Assume that the current date (`current_date`) is {pose_date} and add sufficient information for questions which refer to events that might not have come to pass as of this date.
        
        Use concrete events from the articles, providing necessary context. Do not include any information indicating the question was formed on the current date (`current_date`) or using an article.

        Aim for a diverse, clear, and objective set of questions.
        """,
        "prompt": """
        Consider the following news article: 
        {source_article}

        The reason this news article was chosen is: {article_validation_reason}

        You are to create **multiple** forecasting questions based on the valid news articles provided. Each forecasting question consists of a title, a body, and a resolution. Follow these guidelines closely:

        ## Title Guidelines
        - **Definitive Answers:** Formulate a question that has a clear YES or NO answer based on the article.
        - **Sensitivity:** Exclude references to sensitive topics such as religion, politics, gender, or race.
        - **Clarity:** Be straightforward and precise, avoiding ambiguity.
        - **Resolution Date:** Specify the resolution date as "by {month_name}, {year}?"
        - **Context:** Provide sufficient context if event names may not be clear at the `pose_date`.
        - **Article Usage:** Use "a" instead of "the" to enhance predictability.
        - **Planned Events:** Frame questions about announced but incomplete events as proposals or announcements, explicitly avoiding questions about the completion of these events.

        ## Body Guidelines
        - **Disambiguation:** Be precise and avoid unnecessary details that could influence the resolution.
        - **Context:** Expand only on the question title's date; do not include additional information or dates from the article.
        - **Focus on Relevance:** Include only information that directly supports the question title.
        - **Article Usage:** Use "a" instead of "the" to enhance predictability.

        ## Resolution Guidelines
        - **Binary:** Mark the resolution as True for YES and False for NO.
        - **Stability:** The resolution must remain unchanged by the end of the resolution date.
        - **Definitiveness:** Ensure the resolution can be confirmed as YES or NO based on the article.

        ## General Guidelines
        - **Specific Knowledge:** Avoid relying on specific knowledge that might disadvantage participants.
        - **No Reference to the Article:** Do NOT refer to the article in the question's title and body. The forecaster should not understand that the question was formed using an article.
        - **Named Events:** The question should not refer to any specific named events that you (the AI Agent) may not be aware of. Such events would only be named after the `pose_date`, and forecasters would have no information about them.
        - **Numerical Values:** Use clear thresholds for numerical questions and avoid complex calculations.
        - **Predictability:** Ensure details do not make the question predictable; use reasonable approximations and ambiguity such as rough thresholds and describing the events rather than naming them.
        - **Avoid Overly Specific Questions:** Do not reference more than three distinct entities from the source article.
        - **Do Not Fabricate Information:** Base questions solely on the provided article.

        A forecasting question that adheres to these guidelines is considered "proper." Please generate the questions accordingly.

        To reject an article, you may return the following forecasting question with an empty title and body as the reason for rejection: 
        {example_rejected_fq}

        Here are examples of "proper" forecasting questions: 
            Example 1:
                {example_fq_1}
            Example 2:
                {example_fq_2}
            Example 3:
                {example_fq_3}
        """,
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

    example_rejected_fq = {
        "title": "",
        "body": "<Reason for rejection>",
        "resolution": False,
    }

    @classmethod
    def _article_validation_prompt_and_preface_formation(
        cls, article: dict, end_date: datetime, pose_date: datetime
    ) -> tuple[str, str]:
        """
        Forms the forecasting prompt and preface for validating whether an article can be used for forming FQs.

        Args:
            article (dict): Dictionary containing article information with keys 'title', 'description', and 'content'.
            end_date (datetime): The end date used to set context for the forecasting question.
            pose_date (datetime): The day that the forecaster thinks that the question has been posed at.

        Returns:
            tuple[str, str]: A tuple containing the forecasting preface and prompt as strings.
        """
        formatted_article = {
            "title": article["title"],
            "description": article["description"],
            "content": article["content"],
        }
        forecasting_preface = cls.news_validation_prompt["preface"].format(
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            pose_date=pose_date.strftime("%B %d, %Y"),
        )
        forecasting_prompt = cls.news_validation_prompt["prompt"].format(
            source_article=json.dumps(formatted_article, indent=4),
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
        )

        return forecasting_preface, forecasting_prompt

    @classmethod
    def _rough_fq_generation_prompt_and_preface_formation(
        cls,
        article: dict,
        end_date: datetime,
        pose_date: datetime,
        article_validation_reason: str,
    ) -> tuple[str, str]:
        """
        Forms the forecasting prompt and preface for generating the rough forecasting question.

        Args:
            article (dict): Dictionary containing article information with keys 'title', 'description', and 'content'.
            end_date (datetime): The end date used to set context for the forecasting question.
            pose_date (datetime): The day that the forecaster thinks that the question has been posed at.
            article_validation_reason (str): reason why the article was validated in the earlier step.

        Returns:
            tuple[str, str]: A tuple containing the forecasting preface and prompt as strings.
        """
        formatted_article = {
            "title": article["title"],
            "description": article["description"],
            "content": article["content"],
        }
        forecasting_preface = cls.rough_fq_generation_prompt["preface"].format(
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            pose_date=pose_date.strftime("%B %d, %Y"),
        )
        forecasting_prompt = cls.rough_fq_generation_prompt["prompt"].format(
            source_article=json.dumps(formatted_article, indent=4),
            example_fq_1=json.dumps(cls.example_fq_1, indent=4),
            example_fq_2=json.dumps(cls.example_fq_2, indent=4),
            example_fq_3=json.dumps(cls.example_fq_3, indent=4),
            example_rejected_fq=json.dumps(cls.example_rejected_fq, indent=4),
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            article_validation_reason=article_validation_reason,
        )

        return forecasting_preface, forecasting_prompt

    @classmethod
    def _form_rough_fq_from_llm_return_val(
        cls,
        article: dict,
        generated_stripped_forecasting_questions: ForecastingQuestion_stripped_with_resolution_list,
    ) -> list:
        """
        Forms a list of rough forecasting question data from the LLM-generated stripped forecasting questions.

        Args:
            article (dict): Dictionary containing article information with keys 'title', 'description', 'content', 'url', and 'publishedAt'.
            generated_stripped_forecasting_questions (ForecastingQuestion_stripped_with_resolution_list): The LLM-generated stripped forecasting questions.

        Returns:
            list: A list of dictionaries containing either rejected or accepted rough forecasting questions.
        """
        rough_forecasting_questions = []
        for (
            generated_stripped_forecasting_question
        ) in generated_stripped_forecasting_questions.questions:
            if generated_stripped_forecasting_question.title.strip() == "":
                # Rejected for forming forecasting question
                rough_forecasting_questions.append(
                    {
                        "articleTitle": article["title"],
                        "articleDescription": article["description"],
                        "articleContent": article["content"],
                        "articleUrl": article["url"],
                        "articlePulishedAt": article["publishedAt"],
                        "fqRejectionReason": generated_stripped_forecasting_question.body,
                    }
                )
            else:
                # Formed rough forecasting question data
                rough_forecasting_questions.append(
                    {
                        "articleTitle": article["title"],
                        "articleDescription": article["description"],
                        "articleContent": article["content"],
                        "articleUrl": article["url"],
                        "articlePulishedAt": article["publishedAt"],
                        "fqTitle": generated_stripped_forecasting_question.title,
                        "fqBody": generated_stripped_forecasting_question.body,
                        "fqResolution": generated_stripped_forecasting_question.resolution,
                    }
                )
        return rough_forecasting_questions

    @classmethod
    async def validate_articles_for_fq_generation(
        cls, article: dict, model_name: str, end_date: datetime, pose_date: datetime
    ) -> dict:
        """
        Class method to validate whether the article is good enough to create the forecasting question.

        Args:
            article (dict): Dictionary containing article information with keys 'title', 'description', 'content', 'url', and 'publishedAt'.
            model_name (str): The model used to generate the rough forecasting question.
            end_date (datetime): The end date used to set context for the forecasting question.
            pose_date (datetime): The day that the forecaster thinks that the question has been posed at.

        Returns:
            dict: containing information about whether the article is good enough to create the forecasting questions
        """
        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._article_validation_prompt_and_preface_formation(
            article, end_date, pose_date
        )

        news_article_validation_result: ValidationResult = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ValidationResult,
        )

        return {
            "article": article,
            "validation_result": news_article_validation_result.valid,
            "validation_reasoning": news_article_validation_result.reasoning,
        }

    @classmethod
    async def article_to_rough_forecasting_question(
        cls,
        article_valid_information: dict,
        model_name: str,
        end_date: datetime,
        pose_date: datetime,
    ) -> list[dict]:
        """
        Class method to create rough forecasting question data from a given article asynchronously.

        Args:
            article (dict): Dictionary containing article information with keys 'title', 'description', 'content', 'url', and 'publishedAt'.
            model_name (str): The model used to generate the rough forecasting question.
            end_date (datetime): The end date used to set context for the forecasting question.
            pose_date (datetime): The day that the forecaster thinks that the question has been posed at.

        Returns:
            list[dict]: A list of dictionaries containing either rejected or accepted rough forecasting questions.
        """
        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._rough_fq_generation_prompt_and_preface_formation(
            article_valid_information["article"],
            end_date,
            pose_date,
            article_valid_information["validation_reasoning"],
        )

        generated_stripped_forecasting_questions = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution_list,
        )

        return cls._form_rough_fq_from_llm_return_val(
            article_valid_information["article"],
            generated_stripped_forecasting_questions,
        )

    @classmethod
    def article_to_rough_forecasting_question_sync(
        cls, article: dict, model_name: str, end_date: datetime, pose_date: datetime
    ) -> list[dict]:
        """
        Class method to create rough forecasting question data from a given article synchronously.

        Args:
            article (dict): Dictionary containing article information with keys 'title', 'description', 'content', 'url', and 'publishedAt'.
            model_name (str): The model used to generate the rough forecasting question.
            end_date (datetime): The end date used to set context for the forecasting question.
            pose_date (datetime): The day that the forecaster thinks that the question has been posed at.

        Returns:
            list[dict]: A list of dictionaries containing either rejected or accepted rough forecasting questions.
        """
        raise NotImplementedError
        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._rough_fq_generation_prompt_and_preface_formation(
            article, end_date, pose_date
        )

        generated_stripped_forecasting_questions = answer_sync(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution_list,
        )

        return cls._form_rough_fq_from_llm_return_val(
            article, generated_stripped_forecasting_questions
        )

    @classmethod
    def article_to_rough_forecasting_question_download_path(
        cls,
        start_date: datetime,
        end_date: datetime,
        num_pages: int,
        num_articles: int,
        model_name: str,
    ) -> str:
        """
        File path to save the rough intermediate forecasting question data.

        Args:
            start_date (datetime): Start date for downloading news.
            end_date (datetime): End date for downloading news.
            num_pages (int): Number of pages of news that were downloaded.
            num_articles (int): Number of articles in use.
            model_name (str): The model used to generate the rough forecasting questions.

        Returns:
            str: File path for saving the rough forecasting questions data.
        """
        if num_pages == -1:
            num_pages = "all"
        if num_articles == -1 or num_articles == float("inf"):
            num_articles = "all"

        model_name = model_name.replace("/", "__").replace("\\", "__")
        news_save_file_name = f"rough_fq_using_{model_name}_from_{format_news_range_date(start_date)}_to_{format_news_range_date(end_date)}_num_pages_{num_pages}_num_articles_{num_articles}.jsonl"

        return os.path.join(
            cls.news_api_rough_fq_save_dir,
            news_save_file_name,
        )

    @classmethod
    def validated_news_articles_save_path(
        cls,
        start_date: datetime,
        end_date: datetime,
        num_pages: int,
        num_articles: int,
        model_name: str,
    ) -> str:
        """
        File path to save the validation results for the news articles

        Args:
            start_date (datetime): Start date for downloading news.
            end_date (datetime): End date for downloading news.
            num_pages (int): Number of pages of news that were downloaded.
            num_articles (int): Number of articles in use.
            model_name (str): The model used to generate the rough forecasting questions.

        Returns:
            str: File path for saving news articles data
        """
        if num_pages == -1:
            num_pages = "all"
        if num_articles == -1 or num_articles == float("inf"):
            num_articles = "all"

        model_name = model_name.replace("/", "__").replace("\\", "__")
        news_save_file_name = f"validated_news_articles_using_{model_name}_from_{format_news_range_date(start_date)}_to_{format_news_range_date(end_date)}_num_pages_{num_pages}_num_articles_{num_articles}.jsonl"

        return os.path.join(
            cls.news_api_rough_fq_save_dir,
            news_save_file_name,
        )
