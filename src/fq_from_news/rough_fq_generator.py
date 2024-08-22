import json
import os
from datetime import datetime
from common.llm_utils import answer_sync, answer
from .fq_from_news_datatypes import ForecastingQuestion_stripped_with_resolution_list


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

    preface = """
    You are tasked with generating forecasting (prediction) questions that can be answered with a definitive YES or NO based on past events. Ensure questions are clear, unambiguous, and free from trivial or misleading elements. Avoid sensitive topics such as religion, politics, gender, or race, and refrain from subjective terms like "significant."

    Each question must have a resolution that remains definitive from the current date until the specified resolution date in {month_name}, {year}. Assume the current date is {pose_date}; use concrete events and provide sufficient context if necessary.

    Avoid overly specific or politically charged scenarios. Use generalizable terms unless widely known. For numerical questions, use clear thresholds and keep calculations straightforward.

    Aim for a diverse set of questions covering various topics and regularly review to ensure robustness.
    """

    prompt = """
    A forecasting question consists of the title, the body and the resolution.

    **Guidelines for forecasting questions:**
    
        1. Title
        - **Definitive Answers:** Suggest a question with a clear YES or NO answer based on past events.
        - **Bias Prevention:** Avoid biases and ensure the title is not guessable through heuristics.
        - **Sensitivity:** Exclude references to religion, politics, gender, race, or other sensitive matters.
        - **Clarity:** Be straightforward and precise, avoiding ambiguity.
        - **Resolution Date:** Specify as "by {month_name}, {year}?"
        - **Definitiveness:** Ensure the resolution can be confirmed as YES or NO based on available information.
        - **Scope:** The question must align with the exact wording in the source article.
        - **Context:** Provide sufficient context if event names may not be clear at the `pose_date`.
        - **Article Usage:** Use "a" instead of "the" to enhance predictability.
        - **Planned Events:** Frame questions regarding announced but incomplete events as proposals or announcements.
        - **Sufficient Information:** The title should provide enough context to disambiguate events.

        2. Body
        - **Disambiguation:** Be precise, avoiding unnecessary details that could influence the resolution.
        - **Specific Knowledge:** Avoid relying on specific knowledge that could disadvantage participants.
        - **Context:** Only expand on the title’s date; do not include additional information.
        - **Resolution Date:** Match the resolution date in the body with that in the title.
        - **Article Usage:** Use "a" instead of "the" to maintain a predictive tone.

        3. Resolution
        - **Binary:** Mark the resolution as True for YES and False for NO.
        - **Stability:** The resolution must remain unchanged by the end of the resolution date.
        - **Concrete Events:** Base questions only on concrete events, not opinions.

        4. Additional Guidelines
        - **Quantity:** Generate as many high-quality forecasting questions as possible.
        - **Numerical Values:** Use clear thresholds for numerical questions and avoid complex calculations.
        - **Predictability:** Ensure details do not make the question predictable; use reasonable approximations.
        - **Subjective Terms:** Avoid subjective terms like "significant."
        - **Politically Biased Scenarios:** Exclude politically charged questions.
        - **Avoid Overly Specific Questions:** Do not reference more than three distinct entities from the source article.
        - **Do Not Fabricate Information:** Base questions solely on the provided article content.

    A forecasting question following the above guidelines is said to be "proper".

    Here are examples of "proper" forecasting questions with title and body:
        Example 1:
        ```JSON
        {example_fq_1}
        ```
        Example 2:
        ```JSON
        {example_fq_2}
        ```
        Example 3:
        ```JSON
        {example_fq_3}
        ```
    ---

    A news article consist of the title, description and content.
    {article_description}
    
    **Guidelines for news articles that can be used to form forecasting questions**

        1. **Clarity of Content**  
        - The article must present information in a clear and straightforward manner. If the body of the article is ambiguous or convoluted, making it difficult for a reader or an AI forecaster to understand the context or events, the article should be rejected.

        2. **Definitive Events**  
        - The article should focus on concrete events that have occurred or are planned, allowing for a clear YES or NO answer. Articles that discuss hypothetical scenarios or opinions without clear outcomes should not be used.

        3. **Contextual Relevance**  
        - The article must provide sufficient context regarding the events discussed. If the events mentioned are not clearly defined or lack background information necessary for understanding, the article is not suitable.

        4. **Specificity of Information**  
        - The information within the article should be specific enough to allow for the formation of precise forecasting questions. Vague or overly general content that does not lend itself to clear predictions should be rejected.

        5. **Binary Resolution Potential**  
        - The article must imply a resolution that can be confirmed as TRUE (YES) or FALSE (NO) based on the information provided. If the outcome is uncertain or too subjective, the article should not be used.

        6. **Avoidance of Sensitive Topics**  
        - Articles that touch on sensitive subjects—such as religion, politics, gender, or race—should be avoided to prevent bias and ensure neutrality in the forecasting questions.

        7. **Completeness of Information**  
        - The article must contain enough information to create multiple high-quality forecasting questions. If it lacks sufficient detail or is too brief, it should be rejected.

        8. **Absence of Fabricated Information**  
        - All information used to form forecasting questions must be factual and derived directly from the article. Any article that contains unverifiable or fabricated information is not suitable.

        9. **Numerical Clarity**  
        - If the article includes numerical data, it should present clear thresholds or metrics that can be used to formulate precise forecasting questions. Articles with ambiguous numerical references should be rejected.

        10. **Consistency with Guidelines**  
            - The article must adhere to all the established guidelines for creating forecasting questions. Any deviation from these guidelines may result in rejection.

    A news article following the above guidelines is said to be "complete".

    ---

    Consider the following source article:
    ```JSON
    {source_article}
    ```

    You task is to
    1. Judge whether the source news article is "complete", reject it otherwise.
    2. Judge whether a "proper" forecasting question can be generated from it
    3. Return multiple "proper" forecasting questions if the news article is "complete".

    To reject a article, you may return the following forecasting question with an empty title and body as the reason for rejection as follows:
    ```JSON
    {example_rejected_fq}
    ```
    """

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

    article_description = {
        "title": "The headline or title of the article. - you may use this to get context about the forecasting question that you are going to form.",
        "description": "A description or snippet from the article. - you should use this to form the forecasting question",
        "content": "The unformatted content of the article, where available. This is truncated to 200 chars. - you may use this to form the forecasting question",
    }

    @classmethod
    def _prompt_and_preface_formation(
        cls, article: dict, end_date: datetime, pose_date: datetime
    ) -> tuple[str, str]:
        """
        Forms the forecasting prompt and preface based on the given article and end date.

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
        forecasting_preface = cls.preface.format(
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            pose_date=pose_date.strftime("%B %d, %Y"),
        )
        forecasting_prompt = cls.prompt.format(
            source_article=json.dumps(formatted_article, indent=4),
            example_fq_1=json.dumps(cls.example_fq_1, indent=4),
            example_fq_2=json.dumps(cls.example_fq_2, indent=4),
            example_fq_3=json.dumps(cls.example_fq_3, indent=4),
            article_description=json.dumps(cls.article_description, indent=4),
            example_rejected_fq=json.dumps(cls.example_rejected_fq, indent=4),
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
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
    async def article_to_rough_forecasting_question(
        cls, article: dict, model_name: str, end_date: datetime, pose_date: datetime
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
        ) = cls._prompt_and_preface_formation(article, end_date, pose_date)

        generated_stripped_forecasting_questions = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution_list,
        )

        return cls._form_rough_fq_from_llm_return_val(
            article, generated_stripped_forecasting_questions
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
        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._prompt_and_preface_formation(article, end_date, pose_date)

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
        news_save_file_name = f"rough_fq_using_{model_name}_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}_num_articles_{num_articles}.jsonl"

        return os.path.join(
            cls.news_api_rough_fq_save_dir,
            news_save_file_name,
        )
