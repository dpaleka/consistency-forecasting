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
    You are an expert in generating forecasting (prediction) questions that cannot be answered by a simple or trivial forecasting algorithm leveraging biases.
    Contestants submit models early, and based on the current news feed, you create forecasting questions such that their answers are 
        **definitively** (i.e., confirmed by the news feed) either YES or NO, 
        their answers **cannot** be guessed correctly by dumb algorithms, and,
        the questions themselves do **not** contain any internal biases such as religion.
    You reject questions that do not have such definitive answers or can be guessed correctly by dumb algorithms.
    The models are then fed these questions and judged based on their accuracy in answering YES or NO.
    
    Assume that today's date is {current_date} in %Y-%m-%d format. 
    """

    prompt = """
    The mandatory guidelines for creating forecasting (prediction) questions are:
    1. The questions must have a definitive YES or NO answer based on information from the news article about an event that has already occurred.
    In other words, the article given to you must answer whether the forecasting question will resolve to either YES or NO with complete guarantee and no room for speculation.
    2. The questions must serve as forecasting questions, using recent news articles with clear and recent binary outcomes.
    3. All questions should be framed using "Will" and must specify a resolution date using the current week or month or year. You may switch between then depending on your discretion. Use this year only is the status quo is guaranteed not the change and the resolution of the question remains the same.
    You may refer to the resolution being resolved in the current month as either "this month" or the name of the month followed by the year. Alternate between the two formats - for months as well as years.
    4. The questions should be objective, leaving no room for ambiguity. However, they should not be overly specific such that a dumb algorithm can exploit internal biases to result in the correct answer.

    Examples of questions that should **NOT** be used:
    1. Rejected Question: Will the next United Nations Climate Change Conference result in an agreement on carbon emissions?
    Reason: The terms of the agreement are not specified, multiple agreements can be reached, and it lacks a resolution date.
    2. Rejected Question: Will a new variant of COVID-19 cause global disruptions in the next 12 months?
    Reason: "Disruptions" is subjective, and the resolution date is relative, not based on the publication date.
    3. Rejected Question: Will the next G20 summit lead to a major international agreement on digital currency regulations?
    Reason: "Major" is subjective, the summit date is unspecified, and it lacks a resolution date.
    4. Rejected Question: Will Donald Trump attack the U.S. criminal justice system after his historic guilty verdict?
    Reason: "Attack" is subjective, "historic" is not exact, and it is predictable.
    5. Rejected Question: Will Hunter Biden be arrested following his charges for possession?
    Reason: It is common for arrests to follow charges, making it predictable, and the possession item is unspecified.
    6. Rejected Question: Will a higher court in Pakistan uphold the death sentence of a Christian man by the end of July 2024?
    Reason: This is a very specific question and can be guessed by a simple forecaster owing to religious biases and the overly specific use of religion. Moreover, it is very specific as it refers to the man using "the" rather than "a". There are no forms of such questions that are valid. Do not form questions that have inherent biases such as referring to the religion of a person in certain regions.
    7. Rejected Question: Will Hurricane Beryl make landfall in Tulum by the end of this week?
    Reason: It is overly specific as it refers to the hurricane by name. It is incorrect as the forecaster can guess that the Hurricane already exists, despite it being posed as an event in the future. An alternate question could be: "Will a hurricane make landfall in Tulum this week?" Here the hurricane name has been removed.
    8. Rejected Question: Will the NFL be liable for over $4 billion in damages for violating antitrust laws in distributing out-of-market Sunday afternoon games by the end of this month?
    Reason: It is very specific in why the damages are being paid and the amount being paid. It would let a simple forecaster assume that the event does indeed occur. An alternate question could be: "Will the NFL be liable for damages for violating antitrust laws this month?"
    9. Rejected Question: Will a former president be granted absolute immunity for his core constitutional powers by the end of this month?
    Reason: It is overly specific as it refers to the exact terms for which absolute immunity is being granted. There are no forms of alternate questions that a simple forecaster won't be able to guess.
    10. Rejected Question: Will Tarmo Peltokoski start his term as music director of the Hong Kong Philharmonic in the 2026-27 season?
        Reason: It is very specific as to which Philharmonic the person gets the directorship of. Such specificity would be fine if the person and position in question were very relevant such as things pertaining to geopolitics. An alternate question could be: "Will Tarmo Peltokoski start his term as music director of a major orchestra in the 2026-27 season?"
    11. Rejected Question: Will the far-right party gain a significant number of seats in the French legislative elections this month?
        Reason: The term "significant" is not specific enough and can be interpreted differently by different forecasters, making it ambiguous and unreliable for a clear Yes or No answer.
    12. Rejected Question: Will an appeals court reject Garth Drabinsky's antitrust lawsuit against Actors' Equity this month?
        Reason: The question is overly specific about the parties involved and the nature of the lawsuit, making it predictable and not useful for generating broad-based predictions. There are no forms of such questions that can be made into forecasting questions.
    13. Rejected Question: Will a woman cast her ballot in the second round of the legislative elections in France this month?
        Reason: A woman will indeed cast her ballot as France has universal suffrage. So, a dumb forecaster will also get the answer right.
    14. Rejected Question: Will voters at a Paris polling station be acutely aware of the political situation in France this month?
        Reason: This question is not objective. The term "acutely aware" is subjective and cannot be used to form a forecasting question. 
    15. Rejected Question: Will Steve Bannon report to a federal prison in Connecticut to serve his sentence by the end of this month?
        Reason: The question is specific in the case of the lcoation of the federal prison. The alternate question can be - "Will Steve Bannon report to a federal prison to serve a sentence by the end of this month?"
    16. Rejected Question: Will Archbishop Carlo Maria Vigano be excommunicated by the Vatican this month?
        Reason: This is very specific with regards to the Archbishop being excommunicated. Instead, the question could refer to an archbishop and the reason for excommunication as follows.
        Questions that could instead be formed are "Will an Archbishop be excommunicated by the Vatican this month?" and "Will an Archbishop be excommunicated by the Vatican for refusing to recognize the Pope this year?"
        
    A forecasting question is considered "proper" if it follows the above guidelines. 

    Here are examples of "proper" forecasting questions with title and body
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
    Here, the resolution of the question is marked as `True` if the question resolves to Yes and is marked as `False` if it resolves to No.

    Further, here are some examples of a few articles' contents and the high-level overview of the corresponding question that could be made from it. These are for you understanding and should NOT be used as-is.
    1. Article content: A House Democratic lawmaker has become the first in the party to publicly call for President Joe Biden to step down as the Democratic nominee for president...
    Suggested question: Will there be a House Democratic Leader calling for President Joe Biden to step down as the Democratic nominee for president this month?
    Question resolution: Yes 
    2. Article content: College worth it? Only two-thirds of Americans have confidence in higher education...
    Suggested question: Will more than half of Americans have confidence in higher education system by the end of this Month?
    Question resolution: No 
    3. Article content: Andrew Tate can leave Romania but must stay in EU, court rules
    Suggested question 1: Will Andrew Tate be allowed to leave Romania by the end of this month?
    Question resolution 1: Yes
    Suggested question 1: Will Andrew Tate be allowed to leave the EU by the end of this month?
    Question resolution 1: No

    Here are some general pointers and suggestions:
    1. When the article mentions a specific quantitative value, you can phrase the question around a generic number (preferably rounded) close to it, asking whether the actual number will be above or below this value.
        - For instance, if an article states that a party gained 26 seats in an election, you could frame the question as "Will the party gain more than 30 seats compared to last time?" (answer: No) or "Will the party gain fewer than 30 seats compared to last time?" (answer: Yes). You can create various similar questions.
    2. Aim to generate as many forecasting questions as possible.
    3. Ensure that each prediction question's resolution remains accurate for all foreseeable futures up to the quoted resolution date.
        - This should be done while keeping the resolution date slightly vague to prevent easy guesses by a basic forecaster.
        - **Note**: Always include the current week (or even the current day) as the resolution date in cases where the resolution might change. This is especially important when the article mentions a declaration, promise, or announcement by an entity rather than an actual action.

        
    You must reject the article if you cannot form a "proper" forecasting question from it.
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

    Think carefully, aptly and adequately to either form "proper" forecasting questions from the source article or reject it.
    """

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
    example_fq_3 = {
        "title": "Will the TIME 100 Most Influential Companies of 2024 list actually come out in May as promised?",
        "body": 'This page has just said "2024 HONOREES ANNOUNCED IN MAY" for ages now. I\'ve been checking every day for my market, @/Joshua/what-will-be-time-magazines-100-mos-1ccb89e7e3a1 \n\nThe FAQ says:\n\n[image]This market closes at 11:59 PM PT on Friday, May 31st. If the list is published before market close, resolves YES. If not, resolves NO.',
        "resolution": True,
    }

    example_rejected_fq = {
        "title": "",
        "body": "<Reason for rejection>",
        "resolution": False,
    }

    article_description = {
        "title": "The headline or title of the article. - you may use this to get context on the description",
        "description": "A description or snippet from the article. - you should use this to form the forecasting question",
        "content": "The unformatted content of the article, where available. This is truncated to 200 chars. - you may use this to form the forecasting question",
    }

    def _prompt_and_preface_formation(
        article: dict, start_date: datetime
    ) -> tuple[str, str]:
        formatted_article = {
            "title": article["title"],
            "description": article["description"],
            "content": article["content"],
        }
        forecasting_preface = NewsApiRoughForecastingQuestionGenerator.preface.format(
            current_date=start_date.strftime("%Y-%m-%d")
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
        )

        return forecasting_preface, forecasting_prompt

    def _form_rough_fq_from_llm_return_val(
        article: dict,
        generated_stripped_forecasting_questions: ForecastingQuestion_stripped_with_resolution_list,
    ) -> list:
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
        cls, article: dict, model_name: str, start_date: datetime
    ) -> list[dict]:
        """
        Classmethod to create the rough forecasting question data (which should be later passed through FinalForecastingQuestionGenerator
        to create a ForecastingQuestion instance) using a given article in an async manner.

        :article: The News API downloaded article to be used to create the forecasting question
        :model_name: The model being used to create the rough forecasting question
        :start_date: Used to set context of the current date for the model
        """
        (
            forecasting_preface,
            forecasting_prompt,
        ) = NewsApiRoughForecastingQuestionGenerator._prompt_and_preface_formation(
            article, start_date
        )

        generated_stripped_forecasting_questions = await answer(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution_list,
        )

        return (
            NewsApiRoughForecastingQuestionGenerator._form_rough_fq_from_llm_return_val(
                article, generated_stripped_forecasting_questions
            )
        )

    @classmethod
    def article_to_rough_forecasting_question_sync(
        cls, article: dict, model_name: str, start_date: datetime
    ) -> list[dict]:
        """
        Classmethod to create the rough forecasting question data (which should be later passed through FinalForecastingQuestionGenerator
        to create a ForecastingQuestion instance) using a given article in a sync manner.

        :article: The News API downloaded article to be used to create the forecasting question
        :model_name: The model being used to create the rough forecasting question
        :start_date: Used to set context of the current date for the model
        """
        (
            forecasting_preface,
            forecasting_prompt,
        ) = NewsApiRoughForecastingQuestionGenerator._prompt_and_preface_formation(
            article, start_date
        )

        generated_stripped_forecasting_questions = answer_sync(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution_list,
        )

        return (
            NewsApiRoughForecastingQuestionGenerator._form_rough_fq_from_llm_return_val(
                article, generated_stripped_forecasting_questions
            )
        )

    def article_to_rough_forecasting_question_download_path(
        start_date: datetime,
        end_date: datetime,
        num_pages: int,
        num_articles: int,
        model_name: str,
    ) -> str:
        """
        File path to save the rough intermediate forecasting question data

        :start_date: Start date for downloading news
        :end_date: End date for downloading news
        :num_pages: Number of pages of news that were downloaded
        :num_articles: Number of articles in use
        :model_name: The model being used to create the rough forecasting questions

        :returns: file path for saving
        """
        if num_pages == -1:
            num_pages = "all"
        if num_articles == -1 or num_articles == float("inf"):
            num_articles = "all"

        news_save_file_name = f"rough_fq_using_{model_name}_from_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}_num_pages_{num_pages}_num_articles_{num_articles}.jsonl"

        return os.path.join(
            NewsApiRoughForecastingQuestionGenerator.news_api_rough_fq_save_dir,
            news_save_file_name,
        )
