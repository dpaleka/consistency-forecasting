import json
import os
from datetime import datetime
from common.llm_utils import answer_sync, answer
from common.datatypes import ValidationResult
from common.path_utils import get_src_path
from .fq_from_news_datatypes import ForecastingQuestion_stripped_with_resolution_list


class NewsApiRoughForecastingQuestionGenerator:
    """
    Class with functionality to generate "rough" forecasting questions which may required to be iterated over later
    to prune out questions that do not guidelines for forming FQs such as the Navalny Problem.
    """

    news_api_rough_fq_default_save_dir = os.path.join(
        get_src_path(),
        "data/news_feed_fq_generation/news_api/rough_forecasting_question_data",
    )

    @classmethod
    def set_save_directory(cls, directory_path: str):
        if directory_path is None or len(directory_path.strip()) == 0:
            cls.news_api_rough_fq_save_dir = cls.news_api_rough_fq_default_save_dir
        else:
            cls.news_api_rough_fq_save_dir = directory_path

        # Create the save path directory
        os.makedirs(cls.news_api_rough_fq_save_dir, exist_ok=True)

    news_validation_prompt = {
        "preface": """
You are an AI agent responsible for evaluating news articles to determine their suitability for generating forecasting (prediction) questions that can be answered with a definitive YES or NO. Assess each article against the following criteria to ensure clarity, relevance, and factual accuracy:

1. **Clarity of Content**: Is the information presented clearly and straightforwardly? Reject articles that are overly convoluted or difficult to understand.

2. **Focus on Definitive Events**: Does the article discuss concrete events that have occurred or are planned? Evaluate articles referencing past events based on their clarity and context.

3. **Contextual Relevance**: Does the article provide adequate context for the events discussed? While some background gaps are acceptable, the article should allow for a reasonable understanding of the events.

4. **Specificity of Information**: Is the information detailed enough to formulate precise forecasting questions? Reject articles that are too vague to support clear predictions.

5. **Binary Resolution Potential**: Does the article imply a resolution that can be confirmed as TRUE (YES) or FALSE (NO)? Articles may contain subjective elements but should lead to a binary outcome.

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
**Objective:** Generate forecasting questions that can be definitively answered with YES or NO, based on the provided news articles, while testing a forecaster set in the past.

1. **Forecaster’s Context**: The forecaster’s present date is set to **{pose_date}**, so all questions must be framed as if this is the current date. Although the articles may reference future events, your questions must be phrased in a way that the forecaster cannot detect the actual date of question creation. 

2. **Clarity & Precision**: 
- Each question must be **clear**, **specific**, and **unambiguous**.
- Avoid subjective terms like "significant" or any similar ambiguity.
- Do not reference sensitive topics such as religion, politics, or gender.

3. **No Temporal Hints**: 
- Do **not** include any information or context that implies the question was created after **{pose_date}**.
- Ensure no indication that the article is used to inform the question, keeping the creation date fully hidden.

4. **Resolution Period**: 
- The resolution of each question must remain definitive and applicable from the current date until **{month_name}, {year}**.
- Ensure the question’s outcome is verifiable and binary (YES or NO) during this period.

5. **Context from Articles**: 
- Use **concrete events** from the articles, providing enough background to make the question understandable.
- Ensure questions are diverse, covering a wide range of topics without bias or triviality.

**Goal:** Generate a diverse set of precise and objective forecasting questions that seamlessly align with the forecaster’s assumed timeline without revealing the true creation date or source of the information.
""",
        "prompt": """
**Task:** Based on the provided news article, generate **multiple high quality** forecasting questions that follow these structured guidelines. Each question must consist of a title, body, and resolution. 
    The generated forecasting questions must only be formed using information from the article and **no other extrapolations or inferred information**.    

**News Article:**
- {source_article}

### **Title Guidelines**
- **YES/NO Clarity**: Formulate each question so that it can be definitively answered with a YES or NO, based on the article’s content.
    - **Acceptable**: "Will Tesla launch its electric bike by March 2025?"
    - **Not Acceptable**: "Will Tesla’s electric bike launch be a success?"
  
- **Avoid Sensitive Topics**: Do not reference religion, politics, gender, or race.
    - **Acceptable**: "Will Company A release its new electric vehicle by June 2024?"
    - **Not Acceptable**: "Will a certain country's religious leader make a public statement against polygamy by December 2024?"

- **Direct and Precise**: Titles must be straightforward and unambiguous, avoiding vague terms.
    - **Acceptable**: "Will Apple release a new iPhone model by September 2024?"
    - **Not Acceptable**: "Will Apple make a significant product announcement next year?"
    - **Acceptable**: "Will the company announce the new product before the end of Q2 2024?" (The result is clear by the specified date.)
    - **Not Acceptable**: "Will the company continue to develop the product?" (Outcome is too vague and could change after the specified date.)

- **Resolution Date**: Include a resolution date in the format "by {month_name}, {year}?".
    - **Acceptable**: "Will Company A complete its acquisition of Company B by December 2024?"
    - **Not Acceptable**: "Will Company A complete its acquisition of Company B soon?"

- **Context for Clarity**: Provide enough context if event names may not be clear as of the forecaster’s present date ({pose_date}).
    - **Acceptable**: "Will Microsoft announce a new AI framework at its annual Build event by May 2024?"
    - **Not Acceptable**: "Will a company announce new software at a future conference?" (This is very ambiguous)
    - **Not Acceptable**: "Will Microsoft announce a new AI model called GPT-h at its annual Build event by May 2024?" (This very specific since it refers to the AI Model by name which might not be known at {pose_date})

- **Named Entities**: There is no limit on the number of named entities from the article, but the question should avoid becoming overly specific. Overly specific questions may lead to random correct guesses. For example:
    - **Acceptable**: "Will Microsoft launch its new AI model before December 2024?"
    - **Not Acceptable**: "Will Microsoft announce a new AI model called GPT-h at its annual Build event by May 2024?" (This very specific since it refers to the AI Model by name which might not be known at {pose_date})
    - **Not Acceptable**: "Will Microsoft release its GPT-4.1 model in New York City during its August 2024 event?" (This is too specific)

- **Planned or Announced Events**: For events that are planned but not known at {pose_date}, frame them as proposals or announcements rather than completed facts. Include the event name with sufficient context to avoid ambiguity. For example:
    - **Example 1 (Planned Event)**: If MS Dhoni's injury is unknown at {pose_date} and you get an article talking about him getting well enough to play, phrase the question: "Will MS Dhoni return to play after a major injury in July 2024?"
    - **Example 2 (Announced Event)**: For a major political announcement: "Will the U.N. announce a new climate accord by December 2024?"

### **Body Guidelines**
- **Disambiguation**: Stay focused on the title’s core question without introducing unrelated details that could confuse the resolution.
- **No Extra Information**: Only include relevant context to support the title. Avoid extra details or specific dates from the article unless crucial for clarity.

### **Resolution Guidelines**
- **Binary Outcome**: Resolutions must be clearly marked as True for YES and False for NO.
- **Stable Outcome**: Ensure the resolution remains consistent and unchangeable until the resolution date.

- **Definitiveness**: The resolution must be verifiable based solely on the content of the article.

### **General Guidelines**
- **Avoid Specific Knowledge**: Do not require specialized knowledge that could disadvantage forecasters unfamiliar with niche topics.

- **Base Questions on Article Content**: Ensure that all forecasting questions are directly derived from the information presented in the article. Avoid introducing speculative or inferred details that are not explicitly covered in the article.

    **Examples of Improper Forecasting Questions**:

    1. **Example of a Fabricated Question**:
    - **Article Title**: "Union sues Philadelphia over requirement that city workers return to the office full time"
    - **Article Description**: "A union representing Philadelphia city employees has filed a lawsuit against Mayor Cherelle Parker's requirement for full-time office return starting July 15. The suit was filed by District Council 47 of the American Federation of State, County and Municipal Employees."
    - **Article Content**: "A union representing thousands of Philadelphia city employees has filed a lawsuit to prevent Mayor Cherelle Parker's mandate for a full-time return to the office starting July 15."
    - **Improper Forecasting Question**: "Will a judge block the requirement for Philadelphia city employees to return to the office full time by July 2024?"
    - **Reason for Issue**: This question extends beyond the article’s content by implying a judicial outcome, which is not directly covered. The focus should remain on the union's action, not the potential outcome.

    2. **Example of a Fabricated Question**:
    - **Article Title**: "FDA approves a second Alzheimer's drug that can modestly slow disease"
    - **Article Description**: "U.S. health officials have approved a new Alzheimer’s drug from Eli Lilly that can modestly slow the disease. This is the second drug proven to slow the progression of Alzheimer's."
    - **Article Content**: "The FDA has approved Eli Lilly's new Alzheimer's drug, which is shown to modestly slow the progression of the disease."
    - **Improper Forecasting Question**: "Will Eli Lilly's new Alzheimer's drug receive approval from the European Medicines Agency by July 2024?"
    - **Reason for Issue**: This question introduces details about the European Medicines Agency, which are not covered in the article. The question should be focused solely on the FDA's approval of the drug.

    3. **Example of a Fabricated Question**:
    - **Article Title**: "Black farmers' association calls for Tractor Supply CEO's resignation after company cuts DEI efforts"
    - **Article Description**: "The National Black Farmers Association is calling on Tractor Supply’s president and CEO to step down, days after the rural retailer announced that it would drop most of its corporate diversity and climate advocacy efforts. The resignation demand was made on Tuesday following the company's decision."
    - **Article Content**: "NEW YORK (AP) — The National Black Farmers Association called on Tractor Supply’s president and CEO Tuesday to step down after the rural retailer announced that it would drop most of its corporate diversity and climate advocacy efforts."
    - **Improper Forecasting Question**:"Will Tractor Supply's CEO resign by July 2024?"
    - **Reason for Rejection**: The question extends beyond the scope of the article by assuming an action that is not directly mentioned. The article focuses on the call for the CEO's resignation due to the company's decision to cut DEI efforts, not on the CEO's actual resignation.

- **No Direct References to Article**: Do not mention the article, its source or the present date ({pose_date}) explicitly in either the title or body.
    - **Acceptable**: "Will Apple release a new version of macOS by October 2024?"
    - **Not Acceptable**: "According to the article, will Apple plan to release a new version of macOS by October 2024?"

- **Numerical Questions**: If using numerical thresholds, ensure they are clear, straightforward, and avoid requiring complex calculations. For example:
    - **Acceptable**: "Will the price of crude oil exceed $100 per barrel by August 2024?"
    - **Not Acceptable**: "Will crude oil prices increase significantly?"

- **Prevent Predictability**: Ensure the question is not easily predictable by using reasonable approximations and avoiding excessively detailed context that could reveal too much.
    - **Acceptable**: "Will Company A’s market share increase in Q3 2024?"
    - **Not Acceptable**: "Will Company A’s market share increase by exactly 4.5% in Q3 2024?"

- **Avoid Over-Specificity**: Do not include more than three specific details or entities from the article that could make the question overly specific, leading to easy guesses. However, balance is key. Provide enough specificity for context without making it too easy. For example:
    - **Good Balance**: "Will Tesla's next earnings report show an increase of over 10% in profits by June 2024?"
    - **Overly Specific**: "Will Tesla report a 10.3% increase in profits for Q2 2024 in their June 2024 earnings report?"

- **Do Not Fabricate**: Ensure that all details are based on real information from the article—do not invent or speculate beyond what is presented.

**Proper Forecasting Questions** should strictly align with the information provided in the article and avoid any additional speculation or details not mentioned in the source material.

### **Rejection Criteria**
If the article is unsuitable for generating proper forecasting questions, return a forecasting question with an empty title and body, stating the reason for rejection:
- **Rejection Example**: {example_rejected_fq}

### **Examples of Basic Forecasting Questions Generated from News Articles:**
Below are examples of basic forecasting questions generated from news articles. These questions meet the core requirements of being binary, clear, and linked to verifiable events, but may lack the nuance and context needed for higher-quality forecasts:

- **Example 1**: 
    {example_fq_1}
- **Example 2**: 
    {example_fq_2}
- **Example 3**: 
    {example_fq_3}

### **Examples of High-Quality Forecasting Questions:**
The following are examples of high-quality forecasting questions. They not only meet the basic standards of clarity and verifiability but also include carefully considered context, ensuring they are neither too vague nor overly specific. Each question balances specificity with enough ambiguity to prevent easy predictions while maintaining a stable, verifiable resolution. Strive to achieve this level of quality when generating forecasting questions:

- **Example 1**: 
    {example_high_quality_fq_1}
- **Example 2**: 
    {example_high_quality_fq_2}
- **Example 3**: 
    {example_high_quality_fq_3}
""",
    }

    example_rejected_fq = {
        "title": "",
        "body": "Reason for rejection",
        "resolution": False,
    }

    example_fq_1 = {
        "article": {
            "article_description": "Rudy Giuliani has agreed to a last-minute deal to end his personal bankruptcy case and pay about $400,000 to a \
                financial adviser hired by his creditors. The agreement was filed Wednesday in federal Bankruptcy Court in White Plains, New York. A \
                federal judge t…",
            "article_title": "Rudy Giuliani agrees to deal to end his bankruptcy case, pay creditors' financial adviser $400k",
            "article_content": "Rudy Giuliani has agreed to a last-minute deal to end his personal bankruptcy case and pay about $400,000 to a financial \
                adviser hired by his creditors, avoiding a potential deep-dive into the former… [+4026 chars]",
        },
        "forecasting_question": {
            "title": "Will Rudy Giuliani pay over $300,000 as part of a bankruptcy settlement by July 2024?",
            "body": "This question resolves as YES if, by July 31, 2024, Rudy Giuliani is confirmed to have paid over $300,000 as part of a \
                settlement to end his personal bankruptcy case. The payment must be confirmed by official court filings or public statements \
                and reported by at least two reputable news sources. If no such payment is confirmed by the resolution date, the question resolves as NO.",
            "resolution": True,
        },
    }

    example_fq_2 = {
        "article": {
            "article_description": "The South Carolina Supreme Court has ruled the state's death penalty is legal. All five justices agreed with at \
                least part of the ruling, opening the door to restart executions in a state that hasn’t put an inmate to death since 2011. South Carolina's \
                death p…",
            "article_title": "South Carolina Supreme Court rules state death penalty including firing squad is legal",
            "article_content": "COLUMBIA, S.C. (AP) The South Carolina Supreme Court ruled Wednesday that the states death penalty, which now includes a \
                firing squad as well as lethal injection and the electric chair, is legal.\r\nAl… [+4476 chars]",
        },
        "forecasting_question": {
            "title": "Will the South Carolina Supreme Court overturn the legality of the death penalty by July 2024?",
            "body": "This question resolves as YES if the South Carolina Supreme Court reverses its decision and declares the death penalty \
                illegal by July 31, 2024. The decision must be publicly announced by the court and reported by at least two major news outlets. \
                If no such reversal occurs, the question resolves as NO.",
            "resolution": False,
        },
    }

    example_fq_3 = {
        "article": {
            "article_description": "Chipmaker Intel says it is cutting 20% of its massive workforce — about 15,000 jobs — as it tries to turn its \
                business around to compete with more successful rivals like Nvidia and AMD. The Santa Clara, California-based company said \
                Thursday it is also suspen…",
            "article_title": "Intel to lay off more than 20% of its workforce as it cuts costs to try to turn its business around",
            "article_content": "Chipmaker Intel says it is cutting 20% of its huge workforce about 15,000 jobs as it tries to turn its business \
                around to compete with more successful rivals like Nvidia and AMD.\r\nIn a memo to staff,… [+4460 chars]",
        },
        "forecasting_question": {
            "title": "Will Intel announce layoffs exceeding 15% of its workforce by August 2024?",
            "body": "This question resolves as YES if Intel announces layoffs affecting more than 15% of its workforce by August 31, 2024. \
                The layoffs must be confirmed through official company statements or reports from at least two reputable sources such as The New York \
                Times or Bloomberg. If no such announcement is made, the question resolves as NO.",
            "resolution": True,
        },
    }

    example_high_quality_fq_1 = {
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

    example_high_quality_fq_2 = {
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

    example_high_quality_fq_3 = {
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
            month_name=end_date.strftime("%B"),
            year=end_date.strftime("%Y"),
            pose_date=pose_date.strftime("%B %d, %Y"),
            example_rejected_fq=json.dumps(cls.example_rejected_fq, indent=4),
            example_fq_1=json.dumps(cls.example_fq_1, indent=4),
            example_fq_2=json.dumps(cls.example_fq_2, indent=4),
            example_fq_3=json.dumps(cls.example_fq_3, indent=4),
            example_high_quality_fq_1=json.dumps(
                cls.example_high_quality_fq_1, indent=4
            ),
            example_high_quality_fq_2=json.dumps(
                cls.example_high_quality_fq_2, indent=4
            ),
            example_high_quality_fq_3=json.dumps(
                cls.example_high_quality_fq_3, indent=4
            ),
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
        num_pages_str = "all" if num_pages == -1 else str(num_pages)
        num_articles_str = (
            "all"
            if num_articles == -1 or num_articles == float("inf")
            else str(num_articles)
        )

        model_name_cleaned = model_name.replace("/", "__").replace("\\", "__")

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        directory_structure = os.path.join(
            cls.news_api_rough_fq_save_dir,
            model_name_cleaned,
            f"{start_date_str}_to_{end_date_str}",
            f"num_pages_{num_pages_str}",
            f"num_articles_{num_articles_str}",
        )

        os.makedirs(directory_structure, exist_ok=True)

        news_save_file_name = "rough_fq_data.jsonl"

        return os.path.join(directory_structure, news_save_file_name)

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
        num_pages_str = "all" if num_pages == -1 else str(num_pages)
        num_articles_str = (
            "all"
            if num_articles == -1 or num_articles == float("inf")
            else str(num_articles)
        )

        model_name_cleaned = model_name.replace("/", "__").replace("\\", "__")

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        directory_structure = os.path.join(
            cls.news_api_rough_fq_save_dir,
            model_name_cleaned,
            f"{start_date_str}_to_{end_date_str}",
            f"num_pages_{num_pages_str}",
            f"num_articles_{num_articles_str}",
        )

        os.makedirs(directory_structure, exist_ok=True)

        news_save_file_name = "validated_articles.jsonl"

        return os.path.join(directory_structure, news_save_file_name)
