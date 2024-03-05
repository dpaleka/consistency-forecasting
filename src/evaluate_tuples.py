from forecasters import BasicForecaster
from static_checks import NegationChecker
import jsonlines


basic_forecaster = BasicForecaster()
negation_checker = NegationChecker()

model = "meta-llama/Llama-2-70b-chat-hf"
#model = "gpt-3.5-turbo"

for line in jsonlines.open("negation-gpt-3.5-turbo.jsonl"):
    print("start")
    print(f"line: {line}")
    p_answer = basic_forecaster.call(line["P"], model=model)
    notp_answer = basic_forecaster.call(line["notP"], model=model)
    print(f"question: {line['P']}")
    print(f"p_answer: {p_answer}")
    print(f"notp question: {line['notP']}")
    print(f"notp_answer: {notp_answer}")
    if p_answer is None or notp_answer is None:
        print("Either p_answer or notp_answer is None")
        continue
    print(f"violation {negation_checker.violation({'P': p_answer, 'notP': notp_answer})}")
    print("")
    