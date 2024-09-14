import json
import os
from datetime import datetime
from uuid import uuid4
import pytz
from common.datatypes import ForecastingQuestion
from common.utils import shorten_model_name
from common.llm_utils import answer_sync, answer
from common.path_utils import get_src_path
from .fq_from_news_datatypes import (
    ForecastingQuestion_stripped_with_resolution,
    ForecastingQuestionGroundTruthResolutionStrict,
    ForecastingQuestionGroundTruthResolutionLax,
)
from .date_utils import last_datetime_of_month, format_news_range_date


class NewsApiFinalForecastingQuestionGenerator:
    """
    Class with functionality to take the dicts formed by the RoughForecastingQuestionGenerator class
    and parse them to either accept, reject or improve them and then convert them into ForecastingQuestions
    """

    news_api_final_fq_save_dir = os.path.join(
        get_src_path(),
        "data/news_feed_fq_generation/news_api/final_unverified",
    )
    # Create the save path directory
    os.makedirs(news_api_final_fq_save_dir, exist_ok=True)

    rough_fq_validation_prompt = {
        "preface": """
You are an expert in validating and rephrasing forecasting (prediction) questions based on news articles. A forecasting question consists of a title, body and resolution.

Your task is to ensure that each question adheres to the established guidelines and to enhance the phrasing of valid questions. It is important to note that while we are formulating these questions after knowing the resolutions, the forecaster will assume they are answering them as of {pose_date}. The resolution date for the questions should be set as {month_name}, {year}.

Guidelines to be followed are:

1. **Forecaster’s Context**:
   - The forecaster’s present date is set to **{pose_date}**, so all questions must be framed as if this is the current date. Although the articles may reference future events, your questions must be phrased in a way that the forecaster cannot detect the actual date of question creation.

2. **Clarity & Precision**:
   - Each question must be **clear**, **specific**, and **unambiguous**.
   - Avoid subjective terms like "significant" or any similar ambiguity.
   - Do not reference sensitive topics from religion, politics, or gender.

3. **No Temporal Hints**:
   - Do **not** include any information or context that implies the question was created after **{pose_date}**.
   - Ensure no indication that the article is used to inform the question, keeping the creation date fully hidden.

4. **Resolution Period**:
   - The resolution of each question must remain definitive and applicable from the current date until **{month_name}, {year}**.
   - Ensure the question’s outcome is verifiable and binary (YES or NO) during this period.

5. **Factual Basis**:
	- Questions should be directly supported by the article content and not include fabricated information.
""",
        "prompt": """
You are tasked with the following steps:

1. **Validation**:
   - Check if the forecasting question adheres to the provided guidelines. A question is valid if it aligns with the guidelines.

2. **Rejection**:
   - Reject the question if it violates any guidelines. The rejected form should be: 
     {example_rejected_fq}

3. **Rephrasing**:
   - For valid questions, rephrase them to enhance clarity, specificity, and compliance with the guidelines while retaining the original intent. Do NOT add any new information that wasn't included in the original question.

**Data Provided**:
   - {rough_fq_data_desc}

**High-Quality Forecasting Question Examples**:
    - Example 1: 
        {example_fq_1}
    - Example 2: 
        {example_fq_2}
    - Example 3: 
        {example_fq_3}
    - Example 4: 
        {example_fq_4}
    - Example 5: 
        {example_fq_5}

**Task**:
   - Carefully validate and rephrase the following forecasting question:
     {source_rough_fq_data}
""",
    }

    resolution_checker_prompt_strict = {
        "preface": """
You are an AI agent tasked with verifying the resolution of forecasting questions based solely on the content of a provided news article. Your role is crucial in ensuring that the resolutions are definitive and accurately reflect the information available at the time the question was posed.

When evaluating a forecasting question, keep the following principles in mind:
- The resolution should be based on the factual information present in the news article.
- Your assessment should be made from the perspective of the article's publication date, not any other date.
- Reasonable inferences are acceptable, but do not fabricate details or speculate beyond what is stated in the article.
- Use the `None` option if there is absolutely no information in the article that allows for a reasonable inference of either YES or NO. If the article provides any relevant context or information that can lead to a definitive answer, choose either `True` or `False`.
""",
        "prompt": """
Consider the following news article:
    Title: {article_title}
    Description: {article_description}
    Content: {article_content}
    Date: {article_date}

Now, consider this forecasting question: {question_title}

For additional context, use the following information to disambiguate the question:
    {question_body}

Your task is to determine the resolution of the question based solely on the factual information present in the news article, assuming the article's publication date is the current date. Return:
1. `True` if the answer to the question can be reasonably inferred as YES.
2. `False` if the answer to the question can be reasonably inferred as NO.
3. `None` if there is absolutely no information in the article that allows for a reasonable inference of either YES or NO.

Please provide a brief justification for your answer, citing specific details from the article that support your reasoning.
""",
    }

    resolution_checker_prompt_lax = {
        "preface": """
You are an AI agent tasked with verifying the resolution of forecasting questions based solely on the content of a provided news article. Your role is crucial in ensuring that the resolutions are definitive and accurately reflect the information available at the time the question was posed.

When evaluating a forecasting question, keep the following principles in mind:
- The resolution should be based on the factual information present in the news article.
- Your assessment should be made from the perspective of the article's publication date, not any other date.
- Reasonable inferences are acceptable, but do not fabricate details or speculate beyond what is stated in the article.
- You must provide an answer of either `True` or `False`. If the article does not provide sufficient information to definitively determine an answer, choose the option that aligns more closely with the context or implications presented in the article.
""",
        "prompt": """
Consider the following news article:
    Title: {article_title}
    Description: {article_description}
    Content: {article_content}
    Date: {article_date}

Now, consider this forecasting question: {question_title}

For additional context, use the following information to disambiguate the question:
    {question_body}

Your task is to determine the resolution of the question based solely on the factual information present in the news article, assuming the article's publication date is the current date. Return:
1. `True` if the answer to the question can be reasonably inferred as YES.
2. `False` if the answer to the question can be reasonably inferred as NO.

Please provide a brief justification for your answer, citing specific details from the article that support your reasoning. If you find the information ambiguous, select the answer that best fits the context provided in the article.
""",
    }

    rough_fq_data_desc = {
        "articleTitle": "The title of the news article that was used to form the forecasting question",
        "articleDescription": "The description of the news article that was used to form the forecasting question",
        "articleContent": "The content of the news article that was used to form the forecasting question",
        "fqTitle": "The title of the forecasting question",
        "fqBody": "The body of the forecasting question",
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
        "title": "Will a major cryptocurrency be named after a cricket term by July 2025?",
        "body": 'This question will resolve as Yes if, by 31 July, 2025, a cryptocurrency that is ranked within the top 100 by market \
            capitalization according to a recognized cryptocurrency market analysis platform (e.g., CoinMarketCap, CoinGecko) is named after a \
            cricket term. The term must be widely recognized within the cricket community and must directly relate to the sport (e.g., "Wicket", \
            "Bowler", "Century"). The naming of the cryptocurrency must be intentional, with clear references to its cricket-related origin in its \
            official documentation or announcements by its creators. In the event of multiple cryptocurrencies meeting these criteria, the question will \
            resolve as Yes if at least one of them is within the top 100 by market capitalization. This question resolves as NO if no such cryptocurrency \
            exists by the specified date.',
        "resolution": False,
    }

    example_fq_2 = {
        "title": "Will a Formula 1 Grand Prix be hosted in a country currently under international sanctions by December 2025?",
        "body": 'This question will resolve as Yes if, by December 31, 2025, a Formula 1 Grand Prix is officially announced and \
            scheduled to take place in a country that, at the time of the announcement, is under international sanctions by the United Nations, \
            the European Union, the United States, or any other major international body recognized for imposing sanctions.\n\nFor the purpose of \
            this question, "international sanctions" refer to financial, trade, or other sanctions imposed by international bodies or coalitions\
            of countries against a nation for political, economic, or human rights reasons. The sanctions must be widely reported and recognized by \
            reputable news sources (BBC, The Guardian, New York Times, Washington Post).\n\nIn the event of a Grand Prix being announced in a \
            country that later has sanctions lifted before the race occurs, the question will still resolve as Yes if the sanctions were in place \
            at the time of the announcement. Temporary or partial lifting of sanctions for the event does not affect the resolution.\n\nThis question does \
            not consider unofficial or speculative announcements. Confirmation must come from the Formula 1 organization or the sanctioned country\'s government.',
        "resolution": True,
    }

    example_fq_3 = {
        "title": "Will South Korea become the leader in global digital governance by December 2030?",
        "body": "This question will resolve as Yes if, by December 31, 2030, South Korea is recognized as the global leader in digital governance. \
            Recognition must come from at least two of the following authoritative sources: the United Nations, the World Bank, the Digital Nations \
            (formerly known as the D5), or a consensus among at least three major technology-focused publications (e.g., Wired, TechCrunch, The Verge). \
            Criteria for leadership in digital governance include but are not limited to: - Implementation of advanced digital services across government \
            sectors. - Adoption of cutting-edge technologies in public administration. - Demonstrable impact of digital governance on improving public \
            services and citizen engagement. - Leadership in international digital policy discussions and agreements. In the event of a tie or close \
            competition with another nation, the question will resolve as Yes only if South Korea is clearly distinguished as the leader by the majority \
            of the aforementioned sources. Edge cases, such as temporary leadership positions or recognition in a single aspect of digital governance, \
            do not meet the resolution criteria.",
        "resolution": False,
    }

    example_fq_4 = {
        "title": "Will the NFL host a game in a country currently under international sanctions by November 2025?",
        "body": "This question will resolve as Yes if, by November 30, 2025, the National Football League (NFL) has officially hosted a regular season \
            or playoff game in a country that, at the time of the game, is under international sanctions by the United Nations, the European Union, \
            the United States, or any other major international body. The game must be part of the official NFL season schedule, and not a pre-season \
            or exhibition game. Confirmation must come from an official NFL announcement or credible news reports from at least three major news organizations \
            (e.g., ESPN, BBC, The New York Times, The Washington Post). In the event that sanctions are lifted on a country after the announcement of the game \
            but before the game itself, the question will still resolve as Yes if the sanctions were in place at the time of the announcement. \
            Temporary or partial sanctions specifically targeting individuals or entities within the country do not qualify; the sanctions must be \
            country-wide.",
        "resolution": True,
    }

    example_fq_5 = {
        "title": "Will Russia develop a new vaccine for a global pandemic that is over 90% effective by December 2030?",
        "body": "This question will resolve as Yes if, by December 31, 2030, Russia has developed a new vaccine that is confirmed to be over \
            90% effective against a global pandemic virus. The vaccine's effectiveness must be verified through Phase 3 clinical trial results \
            published in a peer-reviewed medical journal or officially announced by the World Health Organization (WHO) or the Russian Ministry of Health. \
            The pandemic in question must be recognized as such by the WHO at the time of the vaccine's development. In the event of multiple vaccines being \
            developed, only the first to meet these criteria will be considered for resolution. The vaccine must be designed for human use. In the absence \
            of direct confirmation from the WHO or the Russian Ministry of Health, credible reports from at least three major news organizations \
            (BBC, The Guardian, Reuters, Bloomberg, New York Times, Washington Post) are sufficient for resolution.",
        "resolution": False,
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
    def _rough_fq_validation_prompt_and_preface_formation(
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
        forecasting_preface = cls.rough_fq_validation_prompt["preface"].format(
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            pose_date=pose_date.strftime("%B %d, %Y"),
        )
        forecasting_prompt = cls.rough_fq_validation_prompt["prompt"].format(
            source_rough_fq_data=rough_fq_data,
            example_fq_1=json.dumps(cls.example_fq_1, indent=4),
            example_fq_2=json.dumps(cls.example_fq_2, indent=4),
            example_fq_3=json.dumps(cls.example_fq_3, indent=4),
            example_fq_4=json.dumps(cls.example_fq_4, indent=4),
            example_fq_5=json.dumps(cls.example_fq_5, indent=4),
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
        article_date,
        res_unchecked_fq_title,
        res_unchecked_fq_body,
        use_lax,
    ) -> tuple[str, str]:
        """
        Forms the resolution checking forecasting prompt and preface from rough forecasting question data.
        """
        if use_lax:
            unformatted_forecasting_prompt = cls.resolution_checker_prompt_lax
        else:
            unformatted_forecasting_prompt = cls.resolution_checker_prompt_strict

        forecasting_preface = unformatted_forecasting_prompt["preface"]
        forecasting_prompt = unformatted_forecasting_prompt["prompt"].format(
            article_title=article_title,
            article_description=article_description,
            article_content=article_content,
            article_date=article_date,
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
            created_date=pose_date,
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
                "question_created": pose_date.strftime("%Y-%m-%d %H:%M:%S"),
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
        raise NotImplementedError("Use async!")
        (
            forecasting_preface,
            forecasting_prompt,
        ) = cls._rough_fq_validation_prompt_and_preface_formation(
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
        ) = cls._rough_fq_validation_prompt_and_preface_formation(
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
        be_lax_in_resolution_checking: bool,
    ) -> ForecastingQuestion_stripped_with_resolution:
        # If the FQ is already deemed invalid, return None
        if cls.check_if_fq_was_rejected(
            final_resolution_unchecked_forecasting_question
        ):
            return None

        # Form prompt and verify resolution
        processed_rough_fq_data = cls._processed_rough_fq_data(rough_fq_data)
        article_title, article_description, article_content, article_date = (
            processed_rough_fq_data["articleTitle"],
            processed_rough_fq_data["articleDescription"],
            processed_rough_fq_data["articleContent"],
            processed_rough_fq_data["articlePulishedAt"],
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
            article_date,
            res_unchecked_fq_title,
            res_unchecked_fq_body,
            be_lax_in_resolution_checking,
        )

        # Generate fqs where resolution has been specifically checked
        if be_lax_in_resolution_checking:
            generated_resolution: ForecastingQuestionGroundTruthResolutionLax = (
                await answer(
                    prompt=forecasting_prompt,
                    preface=forecasting_preface,
                    model=model_name,
                    response_model=ForecastingQuestionGroundTruthResolutionLax,
                )
            )
        else:
            generated_resolution: ForecastingQuestionGroundTruthResolutionStrict = (
                await answer(
                    prompt=forecasting_prompt,
                    preface=forecasting_preface,
                    model=model_name,
                    response_model=ForecastingQuestionGroundTruthResolutionStrict,
                )
            )

        if (
            generated_resolution.resolution is None
            or generated_resolution.resolution
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
        be_lax_in_resolution_checking: bool,
    ) -> ForecastingQuestion:
        """
        Class method to create the final ForecastingQuestion from rough forecasting question data asynchronously.

        Args:
            rough_fq_data (dict): The rough intermediate forecasting question data.
            model_name (str): The model being used to create the rough forecasting question.
            end_date (datetime): Used to set context of the current date for the model.
            pose_date (datetime): The date assumed to be the knowledge cutoff for the forecaster.
            be_lax_in_resolution_checking (bool): WHether to be lax in resolution checking

        Returns:
            ForecastingQuestion: Validated and possibly modified ForecastingQuestion, or None if the title is empty.
        """

        # Generate the FQs post the initial final check
        final_resolution_unchecked_forecasting_question: ForecastingQuestion_stripped_with_resolution = await cls._rough_fq_to_resolution_unchecked_final_stripped_fq(
            rough_fq_data, model_name, end_date, pose_date
        )

        # Generate the resolution checked (not verified) forecasting questions
        final_resolution_checked_forecasting_question: ForecastingQuestion_stripped_with_resolution = await cls._res_unchecked_to_res_checked_final_stripped_fq(
            rough_fq_data,
            model_name,
            final_resolution_unchecked_forecasting_question,
            be_lax_in_resolution_checking,
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
        be_lax_in_resolution_checking: bool,
    ) -> str:
        """
        File path to save the final forecasting questions.

        Args:
            start_date (datetime): Start date for downloading news.
            end_date (datetime): End date for downloading news.
            num_pages (int): Number of pages of news that were downloaded.
            num_articles (int): Number of articles in use.
            model_name (str): The model being used to create the final forecasting questions.
            be_lax_in_resolution_checking (bool): Whether to be lax in resolution checking

        Returns:
            str: File path for saving the final forecasting questions.
        """
        if num_pages == -1:
            num_pages = "all"
        if num_articles == -1 or num_articles == float("inf"):
            num_articles = "all"

        if be_lax_in_resolution_checking:
            lax_str = "lax_res_checking"
        else:
            lax_str = "strict_res_checking"

        model_name = model_name.replace("/", "__").replace("\\", "__")
        news_save_file_name = f"final_fq__{shorten_model_name(model_name)}_{lax_str}_from_{format_news_range_date(start_date)}_to_{format_news_range_date(end_date)}_num_pages_{num_pages}_num_articles_{num_articles}.jsonl"

        return os.path.join(
            cls.news_api_final_fq_save_dir,
            news_save_file_name,
        )
