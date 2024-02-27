# api = "XXX"
# os.environ["OPENAI_API_KEY"] = api
# comments:
# 1. use .env file and load_dotenv() to hide keys from git
# 2. use llm_utils. read the docs in common/. if you're comfortable with async, rather do async
# client = OpenAI()

import sys

sys.path.append("..")  # ugly hack but we're never going to use this in production

from common.llm_utils import query_api_chat_sync


def negate_simple(question: str) -> str:
    """
    Negates the input question.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. I need you to negate the question provided.  This should be done by adding / removing the word 'not' whenever possible.  Demorgan's laws should be followed with and/or negation.  It should return a question. Avoid using the word won't.",
        },
        {"role": "user", "content": question},
    ]

    response = query_api_chat_sync(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.0,
    )

    return response


def cons_check_template(q, call_func, desc, probs_checker):
    neg = call_func(q)
    probs_checker = checker(q, res)

    return {
        "Callable": call_func,
        "desc": desc,
        "probs_checker": probs_checker,
        "orig_statement": q,
        "negated_statement": neg,
    }
