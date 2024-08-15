import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# from forecaster import Forecaster
# from basic_forecaster import BasicForecaster

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


"""
class ForecastingQuestion(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    body: str
    resolution_date: datetime
    question_type: str
    data_source: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[dict] = None
    resolution: Optional[bool] = None

"""


"""
async def generate_questions_from_question(
    source_question,
    model,
    num_questions,
    source_body=None,
    resolve_by=None,
    similar=None,
):

simliar only for paraphase and consequence (can vary by altering the date)

"""

from consistent_forecaster import ConsistentForecaster


# {"id": "5da935a3-7ffa-4e28-8c1a-5952c89a9a42", "title": "Will USA top the Olympic Medal Table at Paris 2024?", "body": "This question will resolve positively if the United States Olympic Team are the (unique) highest ranked team at the 2024 Paris Olympics. It will resolve ambiguously if the Paris Olympics do not take place before 2027. It will resolve negatively if any team achieves a higher or equal ranking to the US team.\n\nThe medal table is calculated by taking all the medals won by each participating country and ordering by:\n\n1. Number of Gold Medals\n2. (Where 1 is tied) Number of Silver Medals\n3. (Where 2 is tied) Number of Bronze Medals", "resolution_date": "2024-12-31 00:00:00", "question_type": "binary", "data_source": "metaculus", "url": "https://www.metaculus.com/questions/7664", "metadata": {"topics": [], "api_url": "https://www.metaculus.com/api2/questions/7664", "market_prob": 0.82, "resolve_time": "2024-08-10T23:00:00Z", "close_time": "2024-07-25T23:00:00Z", "effected_close_time": null, "background_info": "The 2024 Olympic games is a sporting competition which takes place every four years. It involves a range of different events across multiple sports.\n\nAt the Olympic Games, Gold, Silver and Bronze medals are awarded to 1st, 2nd and 3rd place in each event. (In some events two Bronze medals are awarded mostly combat sports).\n\nThe US team is one of the most successful teams in recent Olympics, topping the medal table in 2020, 2016, 2012, 2004, 2000. Will they repeat that in 2024?\n\n*Related questions*\n\n* [How many medals will Team USA win in Paris 2024?](https://www.metaculus.com/questions/7665/total-medals-won-by-the-usa-at-paris-2024/)\n* [Will France come in the Top 5 at Paris 2024?](https://www.metaculus.com/questions/7669/france-home-game-advantage/"}, "resolution": null}

TEST_1 = {
    "id": "5da935a3-7ffa-4e28-8c1a-5952c89a9a42",
    "title": "Will USA top the Olympic Medal Table at Paris 2024?",
    "body": "This question will resolve positively if the United States Olympic Team are the (unique) highest ranked team at the 2024 Paris Olympics. It will resolve ambiguously if the Paris Olympics do not take place before 2027. It will resolve negatively if any team achieves a higher or equal ranking to the US team.\n\nThe medal table is calculated by taking all the medals won by each participating country and ordering by:\n\n1. Number of Gold Medals\n2. (Where 1 is tied) Number of Silver Medals\n3. (Where 2 is tied) Number of Bronze Medals",
    "resolution_date": "2024-12-31 00:00:00",
    "question_type": "binary",
    "data_source": "metaculus",
    "url": "https://www.metaculus.com/questions/7664",
    "metadata": {
        "topics": [],
        "api_url": "https://www.metaculus.com/api2/questions/7664",
        "market_prob": 0.82,
        "resolve_time": "2024-08-10T23:00:00Z",
        "close_time": "2024-07-25T23:00:00Z",
        "effected_close_time": None,
        "background_info": "The 2024 Olympic games is a sporting competition which takes place every four years. It involves a range of different events across multiple sports.\n\nAt the Olympic Games, Gold, Silver and Bronze medals are awarded to 1st, 2nd and 3rd place in each event. (In some events two Bronze medals are awarded mostly combat sports).\n\nThe US team is one of the most successful teams in recent Olympics, topping the medal table in 2020, 2016, 2012, 2004, 2000. Will they repeat that in 2024?\n\n*Related questions*\n\n* [How many medals will Team USA win in Paris 2024?](https://www.metaculus.com/questions/7665/total-medals-won-by-the-usa-at-paris-2024/)\n* [Will France come in the Top 5 at Paris 2024?](https://www.metaculus.com/questions/7669/france-home-game-advantage/",
    },
    "resolution": None,
}
TEST_1 = ForecastingQuestion(**TEST_1)

TEST_2 = {
    "id": "7a492441-3505-4c98-add6-7fc67a963637",
    "title": "Will Robert F. Kennedy Jr. suspend his 2024 presidential campaign before September 24, 2024?",
    "body": "This question resolves as Yes if before September 24, 2024, Robert F. Kennedy Jr. or his campaign officially announces that he has suspended, terminated, or otherwise ended his 2024 campaign for the office of President of the United States.",
    "resolution_date": "2024-09-23 00:00:00",
    "question_type": "binary",
    "data_source": "metaculus",
    "url": "https://www.metaculus.com/questions/26100",
    "metadata": {
        "topics": [],
        "api_url": "https://www.metaculus.com/api2/questions/26100",
        "market_prob": 0.206,
        "resolve_time": "2024-09-24T14:30:00Z",
        "close_time": "2024-07-13T14:30:00Z",
        "effected_close_time": None,
        "background_info": "[The 2024 United States presidential election](https://en.wikipedia.org/wiki/2024_United_States_presidential_election?useskin=vector) is scheduled to be held on Tuesday, November 5, 2024.\n\nAs of July 10 2024, incumbent Democratic president Joe Biden is seeking re-election and is the [presumptive nominee of his party.](https://apnews.com/article/biden-presumptive-nominee-election-president-democrat-63b66006d4bc45354343228e323e3baa) April five-candidate [polls](https://www.realclearpolling.com/polls/president/general/2024/trump-vs-biden-vs-kennedy-vs-west-vs-stein) have found him winning about 40% of the national popular vote.\n\nRepublican former president Donald Trump is seeking election to a second, non-consecutive term, and is the [presumptive nominee of his party.](https://www.politico.com/news/2024/03/12/donald-trump-clinches-republican-presidential-nomination-00146675) April five-candidate [polls](https://www.realclearpolling.com/polls/president/general/2024/trump-vs-biden-vs-kennedy-vs-west-vs-stein) have found him winning about 42% of the national popular vote.\n\nAttorney and political activist [Robert F. Kennedy Jr.](https://en.wikipedia.org/wiki/Robert_F._Kennedy_Jr.?useskin=vector) sought the Democratic nomination in the 2024 cycle, but in October 2023, switched to [running as an independent](https://apnews.com/article/rfk-jr-presidential-campaign-independent-2024-30d940109c4956de9c81f332ec418463) candidate. In March 2024, [Kennedy named his running mate](https://apnews.com/article/rfk-bobby-kennedy-vp-running-mate-6be6d7e04ba7d9e74190b8c01a1bf075), attorney Nicole Shanahan. Recent five-candidate [polls](https://www.realclearpolling.com/polls/president/general/2024/trump-vs-biden-vs-kennedy-vs-west-vs-stein) have found Kennedy winning about 9.5% of the national popular vote, an unusually high figure for a third-party or independent candidate. Kennedy has also reportedly raised, together with outside groups, [over $98 million](https://www.opensecrets.org/2024-presidential-race) in support of his presidential run, an unusually high figure for a third-party or independent candidate.  \n\nGiven Kennedy's unusually high polling and fundraising figures, he may be in a position to significantly influence the outcome of an otherwise close election, particularly if he were to drop out of the race and encourage his supporters to lend their support to one of the major party candidates.",
    },
    "resolution": None,
}
TEST_2 = ForecastingQuestion(**TEST_2)


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
## extraneou combo ones
#symmetry
# But

#condcond

"""


"""
from static_checks.Checker import (
    AndChecker,
    OrChecker,
    ConsequenceChecker,
    ParaphraseChecker,
    CondCondChecker,
    # Add any other checkers you want to import here
)

"""

from common.datatypes import ForecastingQuestion
from common.llm_utils import answer, answer_sync, Example
from forecaster import Forecaster


"""
## list of questions that just have title
new_questions = asyncio.run(
    generate_questions_from_question(
        TEST_1.title,
        "gpt-4o",
        3,
        source_body=TEST_1.body,
        resolve_by=TEST_1.resolution_date,
        similar=None,
    )
)

print("XXXXXX")
for i in new_questions:
    print(i)
quit()

"""


class PromptedToCons_Forecaster(Forecaster):
    def __init__(
        self,
        preface: str = None,
        checks: list[str] = None,
        examples: list[Example] = None,
    ):
        self.consistency_checks = {
            "neg": "P = 1 - not_P",
            "andor": "P = P_or_Q + P_and_Q - Q",
            "and": "max(P + Q - 1, 0) <= P_and_Q <= min(P, Q)",
            "or": "max(P, Q) <= P_or_Q <= min(1, P + Q)",
            "but": "P = P_or_Q - Q_and_not_P",
            "cond": "P = P_and_Q / Q_given_P",
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

        self.preface = preface or (
            """You are an informed and well-calibrated forecaster. I need you to give me the best probability estimate of the following events or questions resolving YES.\n\n
            The questions can be labelled with "P", "Q", or logical combination of such events, for example "Q_given_P", "not_P", Q_and_P" etc.\n\n
            I will also provide you a list of consistency rules that your probability estimates must obey.\n\n
            The consistency rules ar as follows:\n
            {}\n
            Output your answer in the following way.\n
            For each question label output the corresponding probability rounded to two decimal places.\n
            "P": 0.42\n
            "Q": 0.23\n
            "Q_given_P": 0.14\n
            "not_P": 0.58\n
            "P_and_Q" = 0.21\n
            ...\n
            And so on and so forth. Note there can be more or less labels than the example I provided.\n
            After that, I will again ask you to verify that the probabilities you provided are consistent with the rules I give you. I want you to explicitly go through each rule and evaluate whether your response is consistent with that. If it is not, then you should correct that mistake. Note that changing the probability of one label may affect the consistency of other, so you need to be sure that the probabilities you give are consistent across all of the rules.\n\n
            Exxample 1:\n
            rule: P = 1- not_P.\n
            Your logic: P = 0.42, not_P = 0.58. P + not_P = 1. Therefore, this consistency check passes\n\n
            Example 2:\n
            P= 0.11, not_P = 0.58.\n
            In this case, the consistency check fails. Therefore, you should correct either P, not_P, or both such that it passes.\n
            Note that your correction needs to ENSURE that it still passes other consistency checks too.\n\n
            At the end, I want you to output a final line with your prediction just for P again separated by ***.\n\n
            Example:\n
            #Checks you do\n
            ***\n
            0.42\n\n
            Therefore your output should be of the following structure:\n
            - Outputs of your probability estimates for each question
            - ***
            - Your evaluation of each consistency metric of whether it fulfills the consistency criteria.  Clearly state your reasoning for why it does or does not.
            - ***
            - Outputs of probability estimates given your evaluation of the consistency of your previous estimates.  If there are no consistency violations, you may simply repeat your previous response.  If there are, the new probabilities should obey ALL the consistency measurements (keeping in mind that chanding one probability might have to change another to ensure other consistency rules are also obeyed)
            - ***
            - Output of your probability estimate of P.  This should just be a number and matches the P estimate you just gave.

            """.format(str(self.consistency_checks))
        )
        self.examples = examples or []
        """
        self.examples = examples or [
            Example(
                user=ForecastingQuestion_stripped(
                    title="Will Manhattan have a skyscraper a mile tall by 2030?",
                    body=(
                        "Resolves YES if at any point before 2030, there is at least "
                        "one building in the NYC Borough of Manhattan (based on current "
                        "geographic boundaries) that is at least a mile tall."
                    ),
                ),
                assistant=Prob_cot(
                    chain_of_thought=(
                        "As of 2021, there are no skyscrapers a mile tall. There are also "
                        "no plans to build any mile tall skyscraper in new york. The probability "
                        "is: 0.03"
                    ),
                    prob=0.03,
                ),
            )
        ]"""

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

        cons_test = ConsistentForecaster()
        gen_tuple = cons_test.instantiate_cons_tuples(sentence)

        self.forecasting_questions = {}

        for t in gen_tuple:
            cons_tuple = shallow_dict(t)
            for k, v in cons_tuple.items():
                self.forecasting_questions[k] = v

    def call(self, sentence: ForecastingQuestion, **kwargs) -> float:
        self.generate_all_questions(sentence)

        response = answer_sync(
            prompt=self.forecasting_questions.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type(mode="cot"),
            **kwargs,
        )

        print(response)
        return response.prob

    async def call_async(self, sentence: ForecastingQuestion, **kwargs) -> float:
        response = await answer(
            prompt=self.forecasting_questions.__str__(),
            preface=self.preface,
            examples=self.examples,
            response_model=sentence.expected_answer_type(mode="cot"),
            **kwargs,
        )
        return response.prob

    def dump_config(self):
        return {
            "preface": self.preface,
            "examples": [
                {
                    "user": e.user.model_dump_json(),
                    "assistant": e.assistant.model_dump_json(),
                }
                for e in self.examples
            ],
            "related_questions": self.forecasting_questions.__str__(),
            "checks": self.consistency_checks.__str__(),
        }


test_for = PromptedToCons_Forecaster()


print("1")
print(test_for.call(TEST_1))

print("XXXXXXXXXX")
print(test_for.dump_config())
