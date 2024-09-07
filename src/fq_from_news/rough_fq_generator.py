import json
import os
from datetime import datetime
from common.llm_utils import answer_sync, answer
from common.datatypes import ValidationResult
from common.path_utils import get_src_path
from .fq_from_news_datatypes import ForecastingQuestion_stripped_with_resolution_list
from .date_utils import format_news_range_date


class NewsApiRoughForecastingQuestionGenerator:
    """
    Class with functionality to generate "rough" forecasting questions which may required to be iterated over later
    to prune out questions that do not guidelines for forming FQs such as the Navalny Problem.
    """

    news_api_rough_fq_save_dir = os.path.join(
        get_src_path(),
        "data/news_feed_fq_generation/news_api/rough_forecasting_question_data",
    )
    # Create the save path directory
    os.makedirs(news_api_rough_fq_save_dir, exist_ok=True)

    news_validation_prompt = {
        "preface": """
        You are an AI agent responsible for evaluating news articles to determine their suitability for generating forecasting (prediction) questions that can be answered with a definitive YES or NO. Assess each article against the following criteria to ensure clarity, relevance, and factual accuracy:

        1. **Clarity of Content**: Is the information presented clearly and straightforwardly? Reject articles that are overly convoluted or difficult to understand.

        2. **Focus on Definitive Events**: Does the article discuss concrete events that have occurred or are planned? Evaluate articles referencing past events based on their clarity and context.

        3. **Contextual Relevance**: Does the article provide adequate context for the events discussed? While some background gaps are acceptable, the article should allow for a reasonable understanding of the events.

        4. **Specificity of Information**: Is the information detailed enough to formulate precise forecasting questions? Reject articles that are too vague to support clear predictions.

        5. **Binary Resolution Potential**: Does the article imply a resolution that can be confirmed as TRUE (YES) or FALSE (NO)? Articles may contain subjective elements but should lead to a binary outcome.

        6. **Avoidance of Sensitive Topics**: Does the article steer clear of sensitive subjects like religion, politics, gender, or race? Reject articles that may introduce significant bias.

        7. **Completeness of Information**: Does the article provide sufficient detail to create multiple high-quality forecasting questions? Brief articles are acceptable as long as they contain enough information.

        8. **Numerical Clarity**: If applicable, does the article present clear thresholds or metrics for numerical data? Some ambiguity is acceptable, but numerical references should be understandable.

        9. **Sufficiency for Definitive Resolution**: Does the article provide enough information to formulate forecasting questions that yield definitive resolutions from the current date until the specified resolution date in {month_name}, {year}? Ensure the content supports actionable predictions based on concrete events, assuming the current date is {pose_date}.

        10. **Truncated Information**: Truncated information is NOT a cause for rejection. Accept articles that can form prediction questions, even if they reference past events not covered by the LLM's knowledge.

        An article that meets most of these criteria is considered "complete" and suitable for generating forecasting questions, even if it contains minor ambiguities or references past events that may not be fully known.
        """,
        "prompt": """
        Please evaluate the following news article based on the established criteria for completeness: 
        {source_article}

        Based on your assessment, determine if the article is "complete" and suitable for generating forecasting questions. Provide a brief justification for your decision.
        """,
    }

    rough_fq_generation_prompt = {
        "preface": """
        You are tasked with generating forecasting questions that can be answered with a definitive YES or NO based on the provided news articles. Ensure each question is clear, unambiguous, and free from sensitive topics like religion, politics, or gender. Avoid subjective terms like "significant."

        Questions must have a resolution that remains definitive from the current date until {month_name}, {year}. Assume that the current date (`current_date`) is {pose_date} and provide sufficient information for questions that refer to events that may not have occurred as of this date.
        
        Use concrete events from the articles, providing necessary context. Do not include any information indicating the question was formed on the current date (`current_date`) or using the article.

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
        - **Named Entities:** Include at least one named entity (BUt at most three) from the article to enhance specificity.
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

        Here are examples of "proper" forecasting questions formed using articles:
            Example 1:
                {example_fq_1}
            Example 2:
                {example_fq_2}
            Example 3:
                {example_fq_3}
        """,
    }

    example_fq_1 = {
        "article": {
            "article_description": "Rudy Giuliani has agreed to a last-minute deal to end his personal bankruptcy case and pay about $400,000 to a financial adviser hired by his creditors. The agreement was filed Wednesday in federal Bankruptcy Court in White Plains, New York. A federal judge t…",
            "article_title": "Rudy Giuliani agrees to deal to end his bankruptcy case, pay creditors' financial adviser $400k",
            "article_content": "Rudy Giuliani has agreed to a last-minute deal to end his personal bankruptcy case and pay about $400,000 to a financial adviser hired by his creditors, avoiding a potential deep-dive into the former… [+4026 chars]",
        },
        "forecasting_question": {
            "title": "Will Rudy Giuliani pay over $300,000 to a financial adviser as part of a bankruptcy settlement by July 2024?",
            "body": "This question resolves as YES if, by July 31, 2024, it is confirmed that Rudy Giuliani has paid over $300,000 to a financial adviser as part of a settlement to end his personal bankruptcy case. The payment amount should be over $300,000 to account for potential minor adjustments. The confirmation must come from official court documents or statements from involved parties reported by at least two reputable news sources. If the payment is not made the question resolves as NO.",
            "resolution": True,
        },
    }

    example_fq_2 = {
        "article": {
            "article_description": "The South Carolina Supreme Court has ruled the state's death penalty is legal. All five justices agreed with at least part of the ruling, opening the door to restart executions in a state that hasn’t put an inmate to death since 2011. South Carolina's death p…",
            "article_title": "South Carolina Supreme Court rules state death penalty including firing squad is legal",
            "article_content": "COLUMBIA, S.C. (AP) The South Carolina Supreme Court ruled Wednesday that the states death penalty, which now includes a firing squad as well as lethal injection and the electric chair, is legal.\r\nAl… [+4476 chars]",
        },
        "forecasting_question": {
            "title": "Will the South Carolina Supreme Court overturn the legality of the death penalty by July 2024?",
            "body": "This question resolves as YES if SOuth Carolina's Supreme Court deems the death penalty to be illegal by July 31, 2024. The proposal must be officially annnounced and reported by reputable news sources.",
            "resolution": False,
        },
    }

    example_fq_3 = {
        "article": {
            "article_description": "Chipmaker Intel says it is cutting 15% of its massive workforce — about 15,000 jobs — as it tries to turn its business around to compete with more successful rivals like Nvidia and AMD. The Santa Clara, California-based company said Thursday it is also suspen…",
            "article_title": "Intel to lay off more than 15% of its workforce as it cuts costs to try to turn its business around",
            "article_content": "Chipmaker Intel says it is cutting 15% of its huge workforce about 15,000 jobs as it tries to turn its business around to compete with more successful rivals like Nvidia and AMD.\r\nIn a memo to staff,… [+4460 chars]",
        },
        "forecasting_question": {
            "title": "Will Intel propose to lay off more than 15% of its workforce by August 2024?",
            "body": "This question will resolve as YES if Intel announces laying off more than 15% of its workforce by August 31, 2024. The layoffs must be explicitly announced by Intel through official statements or reports from at least two reputable news organizations such as BBC, The Guardian, New York Times, or Washington Post.",
            "resolution": True,
        },
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
