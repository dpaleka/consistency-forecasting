from dotenv import load_dotenv
from simple_parsing import ArgumentParser
import argparse
import os
import json
from download_from_news_api import parse_date, download_news_from_api
from datetime import datetime
from pydantic import BaseModel
from uuid import uuid4
from common.llm_utils import answer_sync
from common.datatypes import ForecastingQuestion
from common.perscache import register_model_for_cache

load_dotenv()


class ForecastingQuestion_stripped_with_resolution(BaseModel):
    title: str
    body: str
    resolution: bool


register_model_for_cache(ForecastingQuestion_stripped_with_resolution)


class ForecastingQuestion_stripped_with_resolution_list(BaseModel):
    questions: list[ForecastingQuestion_stripped_with_resolution]


register_model_for_cache(ForecastingQuestion_stripped_with_resolution_list)


class RoughForecastingQuestionGenerator:
    """
    Class with functionality to generate "rough" forecasting questions which may required to be iterated over later
    to prune out questions that do not guidelines for forming FQs such as the Navalny Problem.
    """

    news_api_rough_fq_save_dir = (
        "./data/news_feed_fq_generation/rough_forecasting_question_data"
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

    Some other general pointers and suggestions are:
    1. If the article refers to a certain quantitative value, the question may be phrased to refer to a generic number (one with mostltyy 0s) near it and ask whether the actual number will be under or over this value.
    For example, if the article says that a certain party gained 26 seats this time in the election, the question could ask whether the party gains more than 30 seats than last time with the answer resolving to No or whether the party gains less than 30 seats than last time with the answer resolving to Yes. You may many variants for such questions.
    2. Try to form as many forecasting questions as POSSIBLE.
    
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
        formatted_article = {
            "title": article["title"],
            "description": article["description"],
            "content": article["content"],
        }
        forecasting_preface = RoughForecastingQuestionGenerator.preface.format(
            current_date=start_date.strftime("%Y-%m-%d")
        )
        forecasting_prompt = RoughForecastingQuestionGenerator.prompt.format(
            source_article=json.dumps(formatted_article, indent=4),
            example_fq_1=json.dumps(
                RoughForecastingQuestionGenerator.example_fq_1, indent=4
            ),
            example_fq_2=json.dumps(
                RoughForecastingQuestionGenerator.example_fq_2, indent=4
            ),
            example_fq_3=json.dumps(
                RoughForecastingQuestionGenerator.example_fq_3, indent=4
            ),
            article_description=json.dumps(
                RoughForecastingQuestionGenerator.article_description, indent=4
            ),
            example_rejected_fq=json.dumps(
                RoughForecastingQuestionGenerator.example_rejected_fq, indent=4
            ),
        )

        generated_stripped_forecasting_questions = answer_sync(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution_list,
        )

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
            RoughForecastingQuestionGenerator.news_api_rough_fq_save_dir,
            news_save_file_name,
        )


class FinalForecastingQuestionGenerator:
    """
    Class with functionality to take the dicts formed by the RoughForecastingQuestionGenerator class
    and parse them to either accept, reject or improve them and then convert them into ForecastingQuestions
    """

    news_api_final_fq_save_dir = (
        "./data/news_feed_fq_generation/final_forecasting_questions"
    )
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

        generated_stripped_final_forecasting_question = answer_sync(
            prompt=forecasting_prompt,
            preface=forecasting_preface,
            model=model_name,
            response_model=ForecastingQuestion_stripped_with_resolution,
        )

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


def generate_rough_forecasting_data_sync(
    articles_download_path: str,
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
) -> None:
    """
    Wrapper for calling functionality to generate rough forecasting questions data in a sync manner.

    :returns: None
    """
    # check if the save path for the rough forecasting questions already exists
    rough_fq_save_path = RoughForecastingQuestionGenerator.article_to_rough_forecasting_question_download_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )
    if os.path.exists(rough_fq_save_path):
        raise RuntimeError(
            f"The rough forecasting questions data has possibly already been generated at {rough_fq_save_path}! Delete it first"
        )

    num_articles_processed = 0
    with open(articles_download_path, "r") as jsonl_file:
        for line in jsonl_file:
            if num_articles_processed >= num_articles:
                break

            article = json.loads(line.strip())
            rough_forecasting_questions = RoughForecastingQuestionGenerator.article_to_rough_forecasting_question_sync(
                article, rough_fq_gen_model_name, start_date
            )

            num_articles_processed += 1

            # Save the rough forecasting question data
            with open(rough_fq_save_path, "a") as jsonl_file:
                for rough_forecasting_question in rough_forecasting_questions:
                    jsonl_file.write(json.dumps(rough_forecasting_question) + "\n")

    print(f"Rough forecasting question data has been saved to {rough_fq_save_path}")


def generate_final_forecasting_question_sync(
    start_date: datetime,
    end_date: datetime,
    num_pages: int,
    num_articles: int,
    rough_fq_gen_model_name: str,
    final_fq_gen_model_name: str,
) -> None:
    """
    Wrapper for calling functionality to generate final forecasting questions in a sync manner.

    :returns: None
    """
    # check if the save path for the final forecasting questions already exists
    final_fq_save_path = (
        FinalForecastingQuestionGenerator.rough_fq_to_final_fq_download_path(
            start_date,
            end_date,
            num_pages,
            num_articles,
            final_fq_gen_model_name,
        )
    )
    if os.path.exists(final_fq_save_path):
        raise RuntimeError(
            f"The Final forecasting questions are possibly already at at {final_fq_save_path}! Delete it first."
        )

    # Need the rough forecasting questions data to exist
    rough_fq_save_path = RoughForecastingQuestionGenerator.article_to_rough_forecasting_question_download_path(
        start_date,
        end_date,
        num_pages,
        num_articles,
        rough_fq_gen_model_name,
    )
    if not os.path.exists(rough_fq_save_path):
        raise RuntimeError(
            "The rough forecasting question data has not been generated yet! Generate it first."
        )

    with open(rough_fq_save_path, "r") as jsonl_file:
        for line in jsonl_file:
            rough_fq = json.loads(line.strip())
            if "fqRejectionReason" not in rough_fq:
                final_forecasting_question = (
                    FinalForecastingQuestionGenerator.rough_fq_to_final_fq_sync(
                        rough_fq, final_fq_gen_model_name, start_date
                    )
                )

                # Save the rough forecasting question data
                if final_forecasting_question is not None:
                    with open(final_fq_save_path, "a") as jsonl_file:
                        jsonl_file.write(
                            json.dumps(
                                {
                                    "id": str(final_forecasting_question.id),
                                    "title": final_forecasting_question.title,
                                    "body": final_forecasting_question.body,
                                    "resolution": final_forecasting_question.resolution,
                                    "question_type": final_forecasting_question.question_type,
                                    "data_source": final_forecasting_question.data_source,
                                    "url": final_forecasting_question.url,
                                    "resolution_date": final_forecasting_question.resolution_date.strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                    "metadata": final_forecasting_question.metadata,
                                }
                            )
                            + "\n"
                        )

    print(f"Final forecasting questions have been saved to {final_fq_save_path}")


def main(args: argparse.Namespace) -> None:
    """
    Pipeline for generating forecasting question using News API downloaded articles.

    :args: Arguments supplied to the main function

    :returns: None
    """

    # If neither is set, do both
    if (not args.only_gen_rough) and (not args.only_gen_final):
        args.only_gen_rough = args.only_gen_final = True

    # Download the articles (skips if already downloaded)
    articles_download_path = download_news_from_api(
        args.start_date, args.end_date, args.num_pages, os.getenv("NEWS_API_KEY")
    )

    # Set number of articles to be generated to a very large number if using all articles
    if args.num_articles == -1:
        args.num_articles = float("inf")

    # Check whether the user does NOT want to use different models for the two steps
    if args.rough_fq_gen_model_name == "":
        args.rough_fq_gen_model_name = args.model_name
    if args.final_fq_gen_model_name == "":
        args.final_fq_gen_model_name = args.model_name

    if args.sync:
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

    else:
        # TODO - model behvaiour for async behaviour as well
        # NOTE - file writing is not thread safe with async, so I'll need to handle that (ask someone whether this is true)
        raise NotImplementedError(
            "Haven't added functionality for async behaviour yet."
        )


if __name__ == "__main__":
    parser = ArgumentParser()

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
        "--model-name",
        type=str,
        help="Model used for generating the FQ from the articles",
        default="gpt-4o-2024-05-13",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Set to True if FQs should be generated WITHOUT leverging aysnc (parallel) behaviour.",
        default=False,
    )
    parser.add_argument(
        "--only-gen-rough",
        action="store_true",
        help="Set to True if only the intermediate rough forecasting questions should be generated and downloaded.",
        default=False,
    )
    parser.add_argument(
        "--only-gen-final",
        action="store_true",
        help="""
        Set to True if the intermediate rough forecasting questions have already been downloaded 
        and you wish to create the final forecasting questions from them.  
        """,
        default=False,
    )
    parser.add_argument(
        "--rough-fq-gen-model-name",
        type=str,
        help="""
        Overrides the value set by --model-name to use a separate model for generating 
        rough intermediuate forecasting question data.
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

    args = parser.parse_args()
    assert not (
        args.only_gen_final and args.only_gen_rough
    ), "To generate both the intermediate rough forecasting questions and the final ones, provide NO --only-* flags!"
    assert (
        args.num_articles == -1 or args.num_articles > 0
    ), "Set a positive number or -1 for --num-articles!"

    main(args)
