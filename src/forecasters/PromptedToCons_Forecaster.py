import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.datatypes import ForecastingQuestion
from common.utils import shallow_dict
from static_checks.MiniInstantiator import (
    Neg,
    Paraphrase,
    Conditional,
    Consequence,
    And,
    Or,
)
from .consistent_forecaster import ConsistentForecaster
from .cot_forecaster import CoT_multistep_Forecaster


## consistency metrices by priority
# neg
def gen_neg_tuple(og_question):
    neg_instantiator = Neg()

    base_sentences = {
        "P": og_question,
    }

    negated_question = neg_instantiator.instantiate_sync(base_sentences)

    negated_question = negated_question.not_P
    return negated_question


# parapharse
def gen_paraphrase_tuple(og_question):
    paraphrase_instantiator = Paraphrase()

    base_sentences = {
        "P": og_question,
    }

    paraphrased_question = paraphrase_instantiator.instantiate_sync(base_sentences)
    paraphrased_question = paraphrased_question.para_P

    return paraphrased_question


# related_q = generate_questions_from_question(og_question.title, model, 1, source_body = og_question.body, )
# Cond


def gen_cond_tuple(og_question_1, related_question_2):
    cond_instantiator = Conditional()

    base_sentences = {"P": og_question_1, "Q": related_question_2}

    q_given_p = cond_instantiator.instantiate_sync(base_sentences)
    q_given_p = q_given_p.Q_given_P

    return q_given_p


# Consequence
def gen_cons_tuple(og_question, consequence_type="misc"):
    """
    consequence_type can be 'quantity', 'time', or 'misc'
    """
    cons_instantiator = Consequence()

    base_sentences = {
        "P": og_question,
    }

    consequence_question = cons_instantiator.instantiate_sync(
        base_sentences, consequence_type=consequence_type
    )
    return consequence_question

    if isinstance(consequence_question, list):
        # If the instantiate_sync method returns a list, take the first element
        consequence_question = consequence_question[0]

    consequence_question = consequence_question.cons_P

    return consequence_question


# And
def gen_and_tuple(og_question_1, og_question_2):
    and_instantiator = And()

    base_sentences = {
        "P": og_question_1,
        "Q": og_question_2,
    }

    and_question = and_instantiator.instantiate_sync(base_sentences)
    and_question = and_question.P_and_Q

    return and_question


# Or
def gen_or_tuple(og_question_1, og_question_2):
    or_instantiator = Or()

    base_sentences = {
        "P": og_question_1,
        "Q": og_question_2,
    }

    or_question = or_instantiator.instantiate_sync(base_sentences)
    or_question = or_question.P_or_Q

    return or_question


"""
## other ones that need to be implemented in bq_funcss from consistent_forecaster
#symmetry
# But

#condcond

"""

## DEFAULTS, Todo

DEFAULT_PROMPTS = [
    """
    GENERAL: Output your general reasoning and thought process.  Here you can be as detailed as you want, mentioning the reasoning of your predictions and how / why each prediction obeys the given consistency rules.  If there are multiple questions P, Q, you can also make comments on their independence or relationship with each other.""",
    """PROB: Output your probability estimates of each of the variables (P, Q, not_P, etc.).  Here, ONLY output the labels, their associated predictions, and NOTHING ELSE. Your output MUST look and be formatted as follows.
    P: 0.xx,
    not_P: 0.xx,
    P_or_Q: 0.xx,
    ...""",
    """CHECK: Go through each rule in CONSISTENCY RULES and check whether each rule is obeyed with your given predictions.  For each rule, first print the mathematical rule and the associated predicted values associated with it.  Then think VERY carefully about whether the outputs obey the mathematical rule. Then output whether it obeys the rule. At the end, print out a summary on whether all of the rules passed, or which ones failed. Your output MUST look and be formatted as follows.
    neg: P = 1- not_P, EQUATION is EVALUATION, 
    andor: P = P_or_Q + P_and_Q - Q, EQUATION is EVALUATION, 
    and:  max(P + Q - 1, 0) <= P_and_Q <= min(P, Q), EQUATION is EVALUATION,
    ...
    {ALL consistency checks passed!} OR {failed_check_1, failed_check_2 ... consistency checks failed!""",
    """PROB: Now again output your probability estimates of each variable in a dict like format like before, but taking account and correcting any consistency violations that occurred.
        Note that changing the predicted probability of one value to satisfy one consistency rule will also affect consistency rules for others.  It is IMPERATIVE that any changes made still ensure all other consistency checks pass too.
        If there were no violations found, then simply output the same dict again.  Your output MUST look like and be formatted as follows.
        P: 0.xx,
        not_P: 0.xx,
        P_or_Q: 0.xx,
        ...""",
    "FLOAT: Now just output your probability estimate of P and nothing else as a decimal!",
]


EXAMPLE_QUESTIONS_DICT = """

{
    'P': ForecastingQuestion(
        id=UUID('c0f19d62-8229-4bac-a0ef-a737f9e63d3c'),
        title='Will USA top the Olympic Medal Table at Paris 2024?',
        body='This question will resolve positively if the United States Olympic Team are the (unique) highest ranked team at the 2024 Paris Olympics...',
        resolution_date=datetime.datetime(2024, 12, 31, 0, 0, tzinfo=tzutc()),
        question_type='binary',
        data_source='synthetic_inst',
        url=None,
        metadata=None,
        resolution=None
    ),

    'not_P': ForecastingQuestion(
        id=UUID('55e2e903-5b4d-46a6-b07b-f153d1268908'),
        title='Will USA not top the Olympic Medal Table at Paris 2024?',
        body='This question will resolve positively if the United States Olympic Team are not the (unique) highest ranked team at the 2024 Paris Olympics...',
        resolution_date=datetime.datetime(2024, 12, 31, 0, 0, tzinfo=tzutc()),
        question_type='binary',
        data_source='synthetic_inst',
        url=None,
        metadata=None,
        resolution=None
    ),

    'para_P': ForecastingQuestion(
        id=UUID('f3bd5351-bd70-4137-bfd8-a69df6d00633'),
        title='Will the United States lead the Olympic Medal Count at the 2024 Paris Games?',
        body='This inquiry will be answered affirmatively if the United States Olympic Team stands as the (sole) top-ranked team at the Paris Olympics in 2024...',
        resolution_date=datetime.datetime(2024, 12, 31, 0, 0, tzinfo=tzutc()),
        question_type='binary',
        data_source='synthetic_inst',
        url=None,
        metadata=None,
        resolution=None
    ),

    'Q': ForecastingQuestion(
        id=UUID('4c1eed2a-8260-4894-805e-3f7ba99540cc'),
        title="Will armed conflicts between the United States and the People's Republic of China lead to at least 100 deaths before 2050?",
        body="Resolution Criteria\nThis question will resolve as Yes if, at any time between January 1, 2021 to January 1, 2050, either of the following occur...",
        resolution_date=datetime.datetime(2049, 12, 31, 0, 0, tzinfo=TzInfo(UTC)),
        question_type='binary',
        data_source='synthetic_inst',
        url=None,
        metadata=None,
        resolution=None
    ),

    'P_and_Q': ForecastingQuestion(
        id=UUID('46d2e6f3-16c7-42da-887e-6d030509fda5'),
        title="Will both of the following occur: (1) Will the USA top the Olympic Medal Table at Paris 2024? (2) Will armed conflicts between the United States and the People's Republic of China lead to at least 100 deaths before 2050?",
        body="This question will resolve positively if both of the following conditions are met: (1) The United States Olympic Team is the (unique) highest ranked team at the 2024 Paris Olympics...",
        resolution_date=datetime.datetime(2049, 12, 31, 0, 0, tzinfo=TzInfo(UTC)),
        question_type='binary',
        data_source='synthetic_inst',
        url=None,
        metadata=None,
        resolution=None
    ),

    'P_or_Q': ForecastingQuestion(
        id=UUID('9200cff7-3f7b-4e7c-9cca-dc6eb960cf2c'),
        title="Will either of the following occur: (a) Will the USA top the Olympic Medal Table at Paris 2024? (b) Will armed conflicts between the United States and the People's Republic of China lead to at least 100 deaths before 2050?",
        body="This question will resolve positively if either of the following occurs: (1) The United States Olympic Team is the (unique) highest ranked team at the 2024 Paris Olympics...",
        resolution_date=datetime.datetime(2049, 12, 31, 0, 0, tzinfo=TzInfo(UTC)),
        question_type='binary',
        data_source='synthetic_inst',
        url=None,
        metadata=None,
        resolution=False
    ),

    'Q_and_not_P': ForecastingQuestion(
        id=UUID('007be931-08a0-4551-8284-69efa15acf8e'),
        title="Will both of the following occur: (1) Armed conflicts between the United States and the People's Republic of China lead to at least 100 deaths before 2050, and (2) the United States not top the Olympic Medal Table at Paris 2024?",
        body="Resolution Criteria\nThis question will resolve as Yes if both of the following conditions are met:\n1. At any time between January 1, 2021 to January 1, 2050, either of the following occurs regarding armed conflicts between the United States and the People's Republic of China...",
        resolution_date=datetime.datetime(2049, 12, 31, 0, 0, tzinfo=TzInfo(UTC)),
        question_type='binary',
        data_source='synthetic_inst',
        url=None,
        metadata=None,
        resolution=None
    ),

    'Q_given_P': ForecastingQuestion(
        id=UUID('e7897608-d089-4b14-bb45-469c3f152123'),
        title="Given the USA tops the Olympic Medal Table at Paris 2024, will armed conflicts between the United States and the People's Republic of China lead to at least 100 deaths before 2050?",
        body="This question will resolve positively if the United States Olympic Team are the (unique) highest ranked team at the 2024 Paris Olympics...",
        resolution_date=datetime.datetime(2049, 12, 31, 0, 0, tzinfo=TzInfo(UTC)),
        question_type='conditional_binary',
        data_source='synthetic_inst',
        url=None,
        metadata=None,
        resolution=None
    )
}

"""

example_consistent = [
    {"role": "user", "content": "QUESTIONS: {}".format(EXAMPLE_QUESTIONS_DICT)},
    {"role": "assistant", "content": "RECEIVED"},
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[0],
    },
    {
        "role": "assistant",
        "content": "P: Historically the united States has had a pretty strong showing in the Olympics, winning by far the most gold medals.  However, in the past, other countries such as China have also been pretty competitive and even overtook the US in gold medals in 2008.  Overall I would say the probability is pretty high at 0.73\nnot_P: Historically the united States has had a pretty strong showing in the Olympics, winning by far the most gold medals.  However, in the past, other countries such as China have also been pretty competitive and even overtook the US in gold medals in 2008.  Overall I would say the probability is pretty low at 0.27\npara_P: Historically the united States has had a pretty strong showing in the Olympics, winning by far the most gold medals and most medals.  However, in the past, other countries such as China have also been pretty competitive and even overtook the US in gold medals in 2008.  Overall I would say the probability is pretty high at 0.73\nQ: Recently there have been some Geo-political tensions brewing between China and the US regarding technology, Taiwan, and politics.  However, generally we live in a peaceful and interconnected world where people want to generally avoid war.  Accounting for these assumptions, I would guess 0.33\nP_and_Q: I think that these two events (US topping the Paris Olympics and a US - China armed conflict leading to 100 deaths) are mostly independent events.  Therefore we can initially guess this as P * Q = 0.73 * 0.33 = 0.24.  However, perhaps there is some negative correlation with these events where if the US does not top the medal count, China could perceive the country as a little weaker and therefore more likely to provoke militarily.  Therefore we can make a small adjustment to 0.23\nP_or_Q: We can calculate this as the probability that P + Q - P_and_Q, which is 0.83\nQ_and_not_P: I think that these two events (US - China armed conflict leading to 100 deaths and the US not topping Olympics tables) are mostly independent events.  Therefore we can initially guess this as Q * not_P = 0.33 * 0.27 = 0.09.  However, perhaps there is some positive correlation with these events where if the US does not top the medal count, China could perceive the country as a little weaker and therefore more likely to provoke militarily.  Therefore we can make a small adjustment to 0.10\nQ_given_P: I think that these two events (US topping the Paris Olympics and a US - China armed conflict leading to 100 deaths) are mostly independent events.  Therefore as a base case, we can guess Q_given_P is just Q = 0.33.  We can provide a slight adjustment downwards due to there being small negative correlation with these events at 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[1],
    },
    {
        "role": "assistant",
        "content": "P: 0.73\nnot_P: 0.27\npara_P: 0.73\nQ: 0.33\nP_and_Q: 0.23\nP_or_Q: 0.83\nQ_and_not_P: 0.10\nQ_given_P: 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[2],
    },
    {
        "role": "assistant",
        "content": "neg: P = 1 - not_P, 0.73 = 1 - 0.27 is TRUE\nandor: P = P_or_Q + P_and_Q - Q, 0.73 = 0.83 + 0.23 - 0.33 is TRUE\nand: max(P + Q - 1, 0) <= P_and_Q <= min(P, Q), max(0.73, + 0.33 - 1, 0) <= 0.23 <= min(0.73, 0.33) is TRUE\nor: max(P, Q) <= P_or_Q <= min(1, P + Q), max(0.73, 0.33) <= 0.83 <= min(1, 0.73 + 0.33) is TRUE\nbut: P = P_or_Q - Q_and_not_P, 0.73 = 0.83 - 0.10 is TRUE\ncond: P ≈ P_and_Q / Q_given_P, 0.73 ≈ 0.23 / 0.32 is TRUE\npara: P = para_P, 0.73 = 0.73 is TRUE\nALL consistency checks passed!",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[3],
    },
    {
        "role": "assistant",
        "content": "P: 0.73\nnot_P: 0.27\npara_P: 0.73\nQ: 0.33\nP_and_Q: 0.23\nP_or_Q: 0.83\nQ_and_not_P: 0.10\nQ_given_P: 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[4],
    },
    {"role": "assistant", "content": "0.73"},
]


example_inconsistent = [
    {"role": "user", "content": "QUESTIONS: {}".format(EXAMPLE_QUESTIONS_DICT)},
    {"role": "assistant", "content": "RECEIVED"},
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[0],
    },
    {
        "role": "assistant",
        "content": "P: Historically the united States has had a pretty strong showing in the olympics, winning by far the most gold medals.  However, in the past, other countries such as China have also been pretty competitive and even overtook the US in gold medals in 2008.  Overall I would say the probability is pretty high at 0.73\nnot_P: Historically the united States has had a pretty strong showing in the olympics, winning by far the most gold medals.  However, in the past, other countries such as China have also been pretty competitive and even overtook the US in gold medals in 2008.  Overall I would say the probability is pretty low at 0.35\npara_P: Historically the united States has had a pretty strong showing in the olympics, winning by far the most gold medals and most medals.  However, in the past, other countries such as China have also been pretty competitive and even overtook the US in gold medals in 2008.  Overall I would say the probability is pretty high at 0.76\nQ: Recently there have been some geo-political tensions brewing between China and the US regarding technolody, Taiwan, and politics.  However, generally we live in a peaceful and interconnected world where people want to generally avoid war.  Accounting for these assumptions, I would guess 0.33\nP_and_Q: I think that these two events (US topping the Paris olympics and a US - China armed conflict leading to 100 deaths) are mostly independent events.  Therefore we can intially guesss this as P * Q = 0.73 * 0.33 = 0.24.  However, perhaps there is some negative correlation with these events where if the US does not top the medal count, China could perceive the country as a little weaker and therefore more likely to provoke militarily.  Therefore we can make a small adjustment to 0.23\nP_or_Q: I think the odds of either the US topping the Paris olympics or a US - China conflict leading to 100 deaths is 0.71\nQ_and_not_P: I think that these two events (US - China armed conflict leading to 100 deaths and the US not topping olmpics tables) are mostly independent events.  Therefore we can intially guesss this as Q * not_P = 0.33 * 0.35 = 0.12\nQ_given_P: I think that these two events (US topping the Paris olympics and a US - China armed conflict leading to 100 deaths) are mostly independent events.  Therefore as a base case, we can guess Q_given_P is just Q = 0.33.  We can provide a slight adjustment downwards due to there being small negative correlation with these events at 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[1],
    },
    {
        "role": "assistant",
        "content": "P: 0.73\nnot_P: 0.35\npara_P: 0.76\nQ: 0.33\nP_and_Q: 0.23\nP_or_Q: 0.71\nQ_and_not_P: 0.12\nQ_given_P: 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[2],
    },
    {
        "role": "assistant",
        "content": "neg: P = 1 - not_P, 0.73 = 1 - 0.35 is FALSE\nandor: P = P_or_Q + P_and_Q - Q, 0.73 = 0.71 + 0.23 - 0.33 is FALSE\nand: max(P + Q - 1, 0) <= P_and_Q <= min(P, Q), max(0.73 + 0.33 - 1, 0) <= 0.23 <= min(0.73, 0.33) is TRUE\nor: max(P, Q) <= P_or_Q <= min(1, P + Q), max(0.73, 0.33) <= 0.71 <= min(1, 0.73 + 0.33) is FALSE\nbut: P = P_or_Q - Q_and_not_P, 0.73 = 0.71 - 0.12 is FALSE\ncond: P ≈ P_and_Q / Q_given_P, 0.73 ≈ 0.23 / 0.32 is TRUE\npara: P = para_P, 0.73 = 0.76 is FALSE\n\nneg, andor, or, but, para consistency checks failed!",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[3],
    },
    {
        "role": "assistant",
        "content": "P: 0.73\nnot_P: 0.27\npara_P: 0.73\nQ: 0.33\nP_and_Q: 0.23\nP_or_Q: 0.83\nQ_and_not_P: 0.10\nQ_given_P: 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[4],
    },
    {"role": "assistant", "content": "0.73"},
]


example_generic_c = [
    {"role": "user", "content": "QUESTIONS: {}".format(EXAMPLE_QUESTIONS_DICT)},
    {"role": "assistant", "content": "RECEIVED"},
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[0],
    },
    {
        "role": "assistant",
        "content": "P: #Reasons P may or may not happen\nnot_P: #Reasons not_P may or may not happen\npara_P: #Reasons para_P may or may not happen\nQ: #Reasons Q may or may not happen\nP_and_Q: #Reasons P_and_Q may or may not happen\nP_or_Q: #Reasons P_or_Q may or may not happen\nQ_and_not_P: #Reasons Q_and_not_P may or may not happen\nQ_given_P: #Reasons Q_given_P may or may not happen",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[1],
    },
    {
        "role": "assistant",
        "content": "P: 0.73\nnot_P: 0.27\npara_P: 0.73\nQ: 0.33\nP_and_Q: 0.23\nP_or_Q: 0.83\nQ_and_not_P: 0.10\nQ_given_P: 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[2],
    },
    {
        "role": "assistant",
        "content": "neg: P = 1 - not_P, 0.73 = 1 - 0.27 is TRUE\nandor: P = P_or_Q + P_and_Q - Q, 0.73 = 0.83 + 0.23 - 0.33 is TRUE\nand: max(P + Q - 1, 0) <= P_and_Q <= min(P, Q), max(0.73, + 0.33 - 1, 0) <= 0.23 <= min(0.73, 0.33) is TRUE\nor: max(P, Q) <= P_or_Q <= min(1, P + Q), max(0.73, 0.33) <= 0.83 <= min(1, 0.73 + 0.33) is TRUE\nbut: P = P_or_Q - Q_and_not_P, 0.73 = 0.83 - 0.10 is TRUE\ncond: P ≈ P_and_Q / Q_given_P, 0.73 ≈ 0.23 / 0.32 is TRUE\npara: P = para_P, 0.73 = 0.73 is TRUE\nALL consistency checks passed!",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[3],
    },
    {
        "role": "assistant",
        "content": "P: 0.73\nnot_P: 0.27\npara_P: 0.73\nQ: 0.33\nP_and_Q: 0.23\nP_or_Q: 0.83\nQ_and_not_P: 0.10\nQ_given_P: 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[4],
    },
    {"role": "assistant", "content": "0.73"},
]


example_generic_i = [
    {"role": "user", "content": "QUESTIONS: {}".format(EXAMPLE_QUESTIONS_DICT)},
    {"role": "assistant", "content": "RECEIVED"},
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[0],
    },
    {
        "role": "assistant",
        "content": "P: #Reasons P may or may not happen\nnot_P: #Reasons not_P may or may not happen\npara_P: #Reasons para_P may or may not happen\nQ: #Reasons Q may or may not happen\nP_and_Q: #Reasons P_and_Q may or may not happen\nP_or_Q: #Reasons P_or_Q may or may not happen\nQ_and_not_P: #Reasons Q_and_not_P may or may not happen\nQ_given_P: #Reasons Q_given_P may or may not happen",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[1],
    },
    {
        "role": "assistant",
        "content": "P: 0.73\nnot_P: 0.35\npara_P: 0.76\nQ: 0.33\nP_and_Q: 0.23\nP_or_Q: 0.71\nQ_and_not_P: 0.12\nQ_given_P: 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[2],
    },
    {
        "role": "assistant",
        "content": "neg: P = 1 - not_P, 0.73 = 1 - 0.35 is FALSE\nandor: P = P_or_Q + P_and_Q - Q, 0.73 = 0.71 + 0.23 - 0.33 is FALSE\nand: max(P + Q - 1, 0) <= P_and_Q <= min(P, Q), max(0.73 + 0.33 - 1, 0) <= 0.23 <= min(0.73, 0.33) is TRUE\nor: max(P, Q) <= P_or_Q <= min(1, P + Q), max(0.73, 0.33) <= 0.71 <= min(1, 0.73 + 0.33) is FALSE\nbut: P = P_or_Q - Q_and_not_P, 0.73 = 0.71 - 0.12 is FALSE\ncond: P ≈ P_and_Q / Q_given_P, 0.73 ≈ 0.23 / 0.32 is TRUE\npara: P = para_P, 0.73 = 0.76 is FALSE\n\nneg, andor, or, but, para consistency checks failed!",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[3],
    },
    {
        "role": "assistant",
        "content": "P: 0.73\nnot_P: 0.27\npara_P: 0.73\nQ: 0.33\nP_and_Q: 0.23\nP_or_Q: 0.83\nQ_and_not_P: 0.10\nQ_given_P: 0.32",
    },
    {
        "role": "user",
        "content": DEFAULT_PROMPTS[4],
    },
    {"role": "assistant", "content": "0.73"},
]


DEFAULT_EXAMPLES = [example_generic_c, example_generic_i]


class PromptedToCons_Forecaster(CoT_multistep_Forecaster):
    def __init__(
        self,
        model: str,
        preface: str = None,
        checks: list[str] = None,
        examples: list[str] = None,
        user_prompts: list[str] = None,
    ):
        super().__init__(model, preface, examples)

        self.consistency_checks = {
            "neg": "P = 1 - not_P",
            "andor": "P = P_or_Q + P_and_Q - Q",
            "and": "max(P + Q - 1, 0) <= P_and_Q <= min(P, Q)",
            "or": "max(P, Q) <= P_or_Q <= min(1, P + Q)",
            "but": "P = P_or_Q - Q_and_not_P",
            "cond": "P ≈ P_and_Q / Q_given_P",
            # "condcond": "P = (P_and_Q_and_R) / (Q_given_P * R_given_(P_and_Q))",
            # "cons": "P <= cons_P",
            "para": "P = para_P",
        }

        if checks:
            self.consistency_checks = {
                k: self.consistency_checks[k]
                for k in checks
                if k in self.consistency_checks
            }

        DEFAULT_PREFACE = """
            You are an informed and well-calibrated forecaster. You need to give me the best probability estimate of the following events or questions resolving YES rounded to two decimals.
            The questions can be labeled with "P", "Q", or logical combination of such events, for example "Q_given_P", "not_P", Q_and_P" etc.
            I will also provide you with a list of consistency rules that your probability estimates must obey.
            The CONSISTENCY RULES are as follows: {}

            In addition, leading every input may be an additional command.  Depending on what this command is, structure your response in that way.
            QUESTIONS: This will be followed by the dict of questions to evaluate.  You must ONLY output "RECEIVED" and NOTHING ELSE.
            GENERAL: You may output whatever you want.
            PROB: You must ONLY output your probability estimates for each variable given to you previously in the following format: 
                    P: 0.xx
                    not_P: 0.xx
                    ...

            CHECK: You must only output each check from CONSISTENCY RULES and an evaluation of whether the check is passed in the format below.  In the end, comment on whether all checks passed or which ones failed.
                    check_name: mathematical_rule, substitute_variables is evaluation_true_false
            FLOAT: You must ONLY output a float and NOTHING ELSE.""".format(
            str(self.consistency_checks)
        )

        self.preface = preface or DEFAULT_PREFACE
        self.user_prompts = user_prompts or DEFAULT_PROMPTS

        self.examples = examples or DEFAULT_EXAMPLES

    def generate_all_questions(self, sentence: ForecastingQuestion):
        """# alternate way to gen questions
        new_questions = generate_questions_from_question(
            sentence.title,
            "gpt-4o",
            2,
            source_body=sentence.body,
            resolve_by=None,
            similar=None,
        )"""

        cons_test = ConsistentForecaster(self.model)
        gen_tuple = cons_test.instantiate_cons_tuples(sentence)

        self.forecasting_questions = {}

        for t in gen_tuple:
            cons_tuple = shallow_dict(t)
            for k, v in cons_tuple.items():
                self.forecasting_questions[k] = v

        file_path = "prompted_neg_debug.txt"
        if not os.path.exists(file_path):
            open(file_path, "w").close()

        with open(file_path, "a") as f:
            f.write(str(ForecastingQuestion) + "\n")
            f.write(str(self.forecasting_questions) + "\n\n")
        return self.forecasting_questions

    async def generate_all_questions_async(self, sentence: ForecastingQuestion):
        """# alternate way to gen questions
        new_questions = generate_questions_from_question(
            sentence.title,
            "gpt-4o",
            2,
            source_body=sentence.body,
            resolve_by=None,
            similar=None,
        )"""

        cons_test = ConsistentForecaster(self.model)
        gen_tuple = await cons_test.instantiate_cons_tuples_async(sentence)

        self.forecasting_questions = {}

        for t in gen_tuple:
            cons_tuple = shallow_dict(t)
            for k, v in cons_tuple.items():
                self.forecasting_questions[k] = v

        file_path = "prompted_neg_debug.txt"
        if not os.path.exists(file_path):
            open(file_path, "w").close()

        with open(file_path, "a") as f:
            f.write(str(ForecastingQuestion) + "\n")
            f.write(str(self.forecasting_questions) + "\n\n")
        return self.forecasting_questions

    def generate_user_prompts(self, questions_dict):
        first_str = "QUESTIONS: {}".format(questions_dict)

        if self.user_prompts[0][:9] == "QUESTIONS":
            return self.user_prompts

        self.user_prompts.insert(0, first_str)

        return self.user_prompts

    def prep_call(self, ForecastingQuestion):
        self.generate_all_questions(ForecastingQuestion)
        self.generate_user_prompts(self.forecasting_questions)

    async def prep_call_async(self, ForecastingQuestion):
        await self.generate_all_questions_async(ForecastingQuestion)
        self.generate_user_prompts(self.forecasting_questions)

    """
    @costly()
    def call(self, ForecastingQuestion, include_metadata=True, **kwargs):
        self.prep_call(ForecastingQuestion)
        return super().call(self.user_prompts, self.examples, include_metadata)

    @costly()
    def call_async(self, ForecastingQuestion, include_metadata=True, **kwargs):
        self.prep_call(ForecastingQuestion)
        return super().call_async(self.user_prompts, self.examples, include_metadata)"""

    def call(self, ForecastingQuestion, include_metadata=True, **kwargs):
        self.prep_call(ForecastingQuestion)
        result = super().call(self.user_prompts, self.examples, include_metadata)

        return result

    async def call_async(self, ForecastingQuestion, include_metadata=True, **kwargs):
        try:
            await self.prep_call_async(ForecastingQuestion)
            result = await super().call_async(
                self.user_prompts, self.examples, include_metadata
            )
            return result
        except Exception as e:
            raise e

    def dump_config(self):
        return {
            **super().dump_config(),
            # "related_questions": self.forecasting_questions.__str__(),
            "checks": self.consistency_checks.__str__(),
            #'forecasting_questions': make_json_serializable(self.forecasting_questions),
        }


##steps
"""
init
prep_call
call (or call_async)
"""
