# From https://github.com/dannyallover/llm_forecasting/blob/57361d38801bfe9f01cb05093058251c80771fa7/llm_forecasting/utils/data_utils.py#L260


reformat_prompt = (
    """I have questions that need to be transformed for clarity.

Here are some examples:
Example 1:
Before: Who will win the 2022-2023 Premier League? (Leicester City)
After: *Will Leicester City win the 2022-2023 Premier League?*

Example 2:
Before: What coalition will govern Berlin after the 2023 repeat state election? (SPD+Greens)
After: *Will SPD+Greens govern Berlin after the 2023 repeat state election?*

Example 3:
Before: If Republicans win control of the House of Representatives in the 2022 election, who will be the next Majority Whip of the U.S. House of Representatives? (Rep. Jim Banks)
After: *If Republicans win control of the House of Representatives in the 2022 election, will Jim Banks be the next Majority Whip of the U.S. House of Representatives?*

Can you now transform this question for clarity: {question}

Please place stars around the transformed question.

Your output should take the following structure:
Before: {insert the original question}
After: *{insert the transformed question}*""",
    ("QUESTION",),
)


def reformat_metaculus_questions(
    data,
    model_name="gpt-3.5-turbo-1106",
    prompt=reformat_prompt,
):
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
        prompt (tuple of str, optional): Prompt to use for model evaluation.
            Default is PROMPT_DICT["data_cleaning"]["reformat"].

    Returns:
        Modifies the input data in-place, and returns None.
    """

    """
    def find_text_between_stars(text):
        match = re.search(r"\*([^*]+)\*", text)
        return match.group(1) if match else None

    for d in data:
        if "? (" in d["title"]:
            prompt = string_utils.get_prompt(
                prompt[0],
                prompt[1],
                question=d["title"],
            )
            response = model_eval.get_response_from_model(
                model_name=model_name, prompt=prompt
            )
            transformed_title = find_text_between_stars(response)
            if transformed_title:
                d["title"] = transformed_title
    """

    return None
