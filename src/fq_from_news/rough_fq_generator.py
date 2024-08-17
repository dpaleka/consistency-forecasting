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
    You are tasked with generating forecasting (prediction) questions that have a definitive YES or NO answer based on past events. These questions must be clear, unambiguous, and free from trivial or misleading elements. Avoid influences from sensitive matters such as religion, politics, gender, or race, and refrain from using subjective terms like "significant."

    Each question's resolution must remain definitive and unchanged from the current date until the specified resolution date in the month of {month_name}, {year}.

    The forecaster will assume the current date is the `pose_date` which is {pose_date}, so use concrete events to form your questions. If context or event names would not be apparent at the `pose_date`, provide sufficient context to avoid revealing that the question was formed later. The simplest litmus test is to check whether you know of the event through solely your training data.

    Avoid overly specific or politically charged scenarios, and make reasonable approximations to ensure robustness and neutrality in your questions. Use generalizable terms instead of specific names if the event would not be known based solely on the training data without the provided article.

    When creating numerical questions, use clear thresholds and avoid complex calculations that could lead to misinterpretation. Keep the numerical aspects straightforward and easy to understand.

    Aim to create a diverse set of questions covering various topics and events. Regularly review and revise to ensure a robust set of forecasting questions.
    """

    prompt = """
    A forecasting question consists of three main components: the title, the body, and the resolution. A forecasting question is considered "proper" if it adheres to the following guidelines for each component:

    1. **Title**:
    - **Definitive Answers**: The title must suggest a question that has a clear YES or NO answer based on news articles about past events, leaving no room for speculation.
    - **Bias Prevention**: The title must be designed to avoid biases and should not be guessable by simple algorithms exploiting recency, confirmation, anchoring, or availability heuristics.
    - **Avoid Internal Biases**: The title must be free from biases related to religion, politics, gender, race, or other sensitive matters. Avoid politically biased scenarios, such as indictments of former presidents for political reasons.
    - **Clarity and Precision**: The title should be straightforward and precise, avoiding any ambiguity or clues that could lead to obvious answers through pattern recognition or simplistic heuristics. Avoid subjective terms like "significant."
    - **Resolution Date**: The resolution date must be the month and year as "by {month_name}, {year}?". Ensure that the resolution remains accurate for all foreseeable futures up to this date.
    - **Definitiveness**: The resolution must be definitive, meaning that it can be confirmed as either YES or NO based on the news feed. Ensure the answer is not subject to interpretation or speculation.
    - **Scope**: The scope of the question must be exactly verified by the exact words given in the source article. Do not make any questions that are derivatory.

    2. **Body**:
    - **Disambiguation**: The body of the question must be articulated with the highest degree of precision. Avoid any unnecessary details that could influence the resolution, and ensure that there is no detectable link between the "resolution" and the "news."
    - **Avoid Specific Knowledge**: The body should not rely on specific knowledge that could disadvantage certain models or participants. Avoid overly specific details that simple forecasters could exploit. Make reasonable approximations if necessary.
    - **Context**: Ensure that the body of the question does not provide contextual information that could lead to a predictable or simplistic resolution. The body can only expand on the date given in the forecasting question's title and does not have any additional information from the article. Moreover, the resolution date given in the body should use the resolution date given in the question title.
    - **Pose Date Context**: If the context or names of the events being used in the forecasting question wouldn't be apparent or logical at the time of the `pose_date`, add sufficient context to ensure the forecaster does not understand that the question was formed at a time after the `pose_date`. Additionally, if the event would not be known based solely on the training data without the provided article, use generalizable terms instead of specific names.

    3. **Resolution**:
    - **Binary**: The resolution of the question is marked as `True` if the question resolves to Yes and is marked as `False` if it resolves to No.
    - **No Intermediate Changes**: The resolution should not have changed by the end of the resolution date. The resolution must remain correct for all foreseeable futures until the resolution date of the month and year.
    - **Concrete Events**: Form questions only from concrete events that have occurred and not from opinions or proclamations. Ensure no possibility of the question resolving between the current date and the resolution date's month.

    **Additional Guidelines**:
    - **Quantity**: Create as many high-quality forecasting questions as possible while adhering to the above criteria.
    - **Numerical Values**: When creating numerical questions, use clear thresholds and avoid complex calculations that could lead to misinterpretation. Keep the numerical aspects straightforward and easy to understand. However, do not use numerical values if the article does not include any such values. 
    - **Avoiding Predictability**: The specificity of details should not make the question predictable. Make reasonable approximations to avoid such issues.
    - **Avoid Subjective Terms**: Ensure that the title and body do not use subjective terms like "significant," which can lead to ambiguity.
    - **Avoid Politically Biased Scenarios**: Avoid questions related to politically biased scenarios, such as indictments or actions of former presidents for political reasons. Use concrete events and avoid politically sensitive topics.
        
    **Examples of Questions That Should NOT Be Used**:

    1. **Rejected Question**: Will the next United Nations Climate Change Conference result in an agreement on carbon emissions?
    - **Reason**: The terms of the agreement are vague, multiple agreements could be reached, and it lacks a resolution date.

    2. **Rejected Question**: Will a new variant of COVID-19 cause global disruptions in the next 12 months?
    - **Reason**: "Disruptions" is subjective, and the resolution date is relative, not based on the publication date.

    3. **Rejected Question**: Will the next G20 summit lead to a major international agreement on digital currency regulations?
    - **Reason**: "Major" is subjective, the summit date is unspecified, and it lacks a resolution date.

    4. **Rejected Question**: Will Donald Trump attack the U.S. criminal justice system after his historic guilty verdict?
    - **Reason**: "Attack" is subjective, "historic" is not exact, and it is predictable.

    5. **Rejected Question**: Will Hunter Biden be arrested following his charges for possession?
    - **Reason**: It is common for arrests to follow charges, making it predictable, and the possession item is unspecified.

    6. **Rejected Question**: Will a higher court in Pakistan uphold the death sentence of a Christian man by the end of <month> 2024?
    - **Reason**: The question is overly specific and biased due to religious references. The specific use of "the" rather than "a" makes it too detailed.

    7. **Rejected Question**: Will Hurricane Beryl make landfall in Tulum by the end of this week?
    - **Reason**: The question is overly specific due to the hurricane's name, making it predictable. An alternate question could be: "Will a hurricane make landfall in Tulum by <month>, 2024?"

    8. **Rejected Question**: Will the NFL be liable for over $4 billion in damages for violating antitrust laws in distributing out-of-market Sunday afternoon games by the end of <month>, 2024?
    - **Reason**: The specificity regarding the amount and reason for damages makes it too predictable. An alternate question could be: "Will the NFL be liable for damages for violating antitrust laws this month?"

    9. **Rejected Question**: Will a former president be granted absolute immunity for his core constitutional powers by the end of July?
    - **Reason**: The question is overly specific regarding the terms of immunity, making it predictable. Moreover, the resolution year is not defined. 

    10. **Rejected Question**: Will Tarmo Peltokoski start his term as music director of the Hong Kong Philharmonic in the July 2024 session?
        - **Reason**: The specificity of the Philharmonic makes it too detailed. An alternate question could be: "Will Tarmo Peltokoski start his term as music director of a major orchestra in the July 2024 session"

    11. **Rejected Question**: Will the far-right party gain a significant number of seats in the French legislative elections this month?
        - **Reason**: "Significant" is subjective and could be interpreted differently, making it ambiguous. Moreover, it says "this month" rather than the concrete resolution date as the exact month and year. 

    12. **Rejected Question**: Will an appeals court reject Garth Drabinsky's antitrust lawsuit against Actors'  by July 2024?
        - **Reason**: The specificity about the parties involved and the nature of the lawsuit makes it too predictable.

    13. **Rejected Question**: Will a woman cast her ballot in the second round of the legislative elections in France this month?
        - **Reason**: It is predictable as women will cast ballots due to universal suffrage. Moreover, it says "this month" rather than the concrete resolution date as the exact month and year. 

    14. **Rejected Question**: Will voters at a Paris polling station be acutely aware of the political situation in France by July 2024?
        - **Reason**: "Acute awareness" is subjective and cannot be objectively measured.

    15. **Rejected Question**: Will Steve Bannon report to a federal prison in Connecticut to serve his sentence by the end of by July 2024?
        - **Reason**: The specificity of the prison location makes it predictable. An alternate question could be: "Will Steve Bannon report to a federal prison to serve a sentence by July 2024?"

    16. **Rejected Question**: Will Archbishop Carlo Maria Vigano be excommunicated by the Vatican this month?
        - **Reason**: The specificity of the Archbishop makes it too detailed. Alternative questions could be: "Will an Archbishop be excommunicated by the Vatican by July 2024?"

    17. **Rejected Question**: "Will the number of fatalities from the propane tank explosion in Izmir, Turkey, increase to more than 5 by July 2024?"
        - *Reason**: The question refers to "the" propane tank explosion which lets the forecaster understand that the event has already taken place.  
        
    **Create as many high-quality forecasting questions as possible, ensuring they meet all criteria outlined. The goal is to generate questions that are objective, challenging, and free from biases while remaining clear and definitive in their expected outcomes.**


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

    **You must reject the article if you cannot form a "proper" forecasting question from it.**
    To reject a article, you may return the following forecasting question with an empty title and body as the reason for rejection as follows:
    ```JSON
    {example_rejected_fq}
    ```

    For this task, the format of the news article given to you will be:
    ```JSON
    {article_description}
    ```

    Generate a "proper" forecasting questions from the following source article. Reject it if you cannot generate a "proper" forecasting question. In case of rejection, you only return one forecasting question. 
    ```JSON
    {source_article}
    ```

    Think carefully, aptly and adequately to either form "proper" forecasting questions (multiple if possible) from the source article or reject it.
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

    def _prompt_and_preface_formation(
        article: dict, end_date: datetime, pose_date: datetime
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
        forecasting_preface = NewsApiRoughForecastingQuestionGenerator.preface.format(
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            pose_date=pose_date.strftime("%B %d, %Y"),
        )
        forecasting_prompt = NewsApiRoughForecastingQuestionGenerator.prompt.format(
            source_article=json.dumps(formatted_article, indent=4),
            example_fq_1=json.dumps(
                NewsApiRoughForecastingQuestionGenerator.example_fq_1, indent=4
            ),
            example_fq_2=json.dumps(
                NewsApiRoughForecastingQuestionGenerator.example_fq_2, indent=4
            ),
            example_fq_3=json.dumps(
                NewsApiRoughForecastingQuestionGenerator.example_fq_3, indent=4
            ),
            article_description=json.dumps(
                NewsApiRoughForecastingQuestionGenerator.article_description, indent=4
            ),
            example_rejected_fq=json.dumps(
                NewsApiRoughForecastingQuestionGenerator.example_rejected_fq, indent=4
            ),
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
        )

        return forecasting_preface, forecasting_prompt

    def _form_rough_fq_from_llm_return_val(
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

    def article_to_rough_forecasting_question_download_path(
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
            NewsApiRoughForecastingQuestionGenerator.news_api_rough_fq_save_dir,
            news_save_file_name,
        )
