# Inspired by https://github.com/dannyallover/llm_forecasting/blob/57361d38801bfe9f01cb05093058251c80771fa7/llm_forecasting/utils/data_utils.py#L260
from common.datatypes import register_model_for_cache
from common.llm_utils import query_api_chat_native, query_api_chat
import asyncio

reformat_system_msg = {
    "role": "system",
    "content": """\
I have questions that need to be transformed for clarity. I will first provide examples of the desired output.

If the question does not need to be transformed, just return the question in the same format.
""",
}

examples = [
    {
        "role": "user",
        "content": """\
Title: Who will win the 2022-2023 Premier League? (Leicester City) """,
    },
    {
        "role": "assistant",
        "content": """\
Title: *Will Leicester City win the 2022-2023 Premier League?* """,
    },
    {
        "role": "user",
        "content": """\
Title: Will SPD+Greens govern Berlin after the 2023 repeat state election?
Body: This question will resolve as **Yes** if the SPD and Greens have a majority in the Berlin state parliament after the 2023 repeat state election. All other coalitions will resolve as **No**.""",
    },
    {
        "role": "assistant",
        "content": """\
Title: Will SPD+Greens govern Berlin after the 2023 repeat state election?
Body: This question will resolve as **Yes** if the SPD and Greens have a majority in the Berlin state parliament after the 2023 repeat state election. All other coalitions will resolve as **No**.""",
    },
    {
        "role": "user",
        "content": """\
Title: Which party will form the government after the next Indian general election in 2024? (BJP)
Body: This question will resolve as **Yes** for the party who has a member sworn in as the next Indian Prime Minister following the next general elections.  All other parties will resolve as **No**.\n\nThis question will resolve for the next general election occuring after June 1, 2022 (currently expected for May 2024)""",
    },
    {
        "role": "assistant",
        "content": """\
Title: Will BJP form the government after the next Indian general election in 2024?
Body: This question will resolve as **Yes** if the BJP has a member sworn in as the next Indian Prime Minister following the next general elections.  All other parties will resolve as **No**.\n\nThis question will resolve for the next general election occuring after June 1, 2022 (currently expected for May 2024)""",
    },
    {
        "role": "user",
        "content": """\
Title: If Republicans win control of the House of Representatives in the 2022 election, who will be the next Majority Whip of the U.S. House of Representatives? (Rep. Jim Banks)
Body: This question will resolve N/A if Republicans do not win control of the House of Representatives in the 2022 election. If they do: This question will resolve as **Yes** for the candidate who is sworn in as the next Majority Whip of the U.S. House of Representatives.  All other candidates will resolve as **No**.""",
    },
    {
        "role": "assistant",
        "content": """\
Title: If Republicans win control of the House of Representatives in the 2022 election, will Jim Banks be the next Majority Whip of the U.S. House of Representatives?
Body: This question will resolve N/A if Republicans do not win control of the House of Representatives in the 2022 election. If they do: This question will resolve as **Yes** if Jim Banks is sworn in as the next Majority Whip of the U.S. House of Representatives. Any other person being sworn in as the Majority Whip will resolve as **No**.""",
    },
]


from pydantic import BaseModel


class ForecastingQuestion_title_body(BaseModel):
    title: str
    body: str | None


register_model_for_cache(ForecastingQuestion_title_body)


async def reformat_metaculus_question(
    title: str, body: str | None, model="gpt-4o-mini"
) -> dict[str, str | None | bool]:
    """
    Reformat questions from Metaculus to be more readable.

    In particular, some questions have a title that ends with a parenthesis,
    containing the actual subject.
    This function rephrases it to be a Yes/No question.

    For example,
    >>> "Who will win the 2020 US presidential election? (Biden)"
    will be reformatted by the langauge model to
    >>> "Will Biden win the 2020 US presidential election?"

    Args:
        data (list of dict): List of questions in dictionary format.
        model_name (str, optional): Language model name, default is
            "gpt-3.5-turbo-1106".

    """

    did_change = False
    if title.endswith(")") and "? (" in title:
        messages = (
            [reformat_system_msg]
            + examples
            + [
                {
                    "role": "user",
                    "content": f"Title: {title}\nBody: {body}"
                    if body is not None
                    else f"Title: {title}",
                }
            ]
        )

        native_response = await query_api_chat_native(
            messages=messages,
            model=model,
        )

        messages += [
            {"role": "assistant", "content": native_response},
            {
                "role": "user",
                "content": """\
Now reformat the title and the body into a Pydantic BaseModel with `title` and `body` fields. If there is no body, leave it empty in your response.""",
            },
        ]

        reformatted_response = await query_api_chat(
            messages=messages,
            model=model,
            response_model=ForecastingQuestion_title_body,
        )
        if reformatted_response.body == "":
            reformatted_response.body = None

        if reformatted_response.title != title or reformatted_response.body != body:
            did_change = True

        return {
            "title": reformatted_response.title,
            "body": reformatted_response.body,
            "did_change": did_change,
        }
    else:
        return {"title": title, "body": body, "did_change": False}


def reformat_metaculus_question_sync(
    title: str, body: str, model="gpt-4o-mini"
) -> dict[str, str | None | bool]:
    return asyncio.run(reformat_metaculus_question(title, body, model))
