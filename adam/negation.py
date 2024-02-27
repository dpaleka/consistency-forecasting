from openai import OpenAI
import os
api = 'XXX'
os.environ["OPENAI_API_KEY"] = api
client = OpenAI()



def negate(question):

    messages=[
        {"role": "system", "content": "You are a helpful assistant. I need you to negate the question provided.  This should be done by adding / removing the word 'not' whenever possible.  Demorgan's laws should be followed with and/or negation.  It should return a question. Avoid using the word won't."},
        {'role': 'user', 'content':question},
        ]


    response = client.chat.completions.create(
      model="gpt-4-1106-preview",
      messages=messages
    )
    return response.choices[0].message.content



def cons_check_template(q, call_func, desc, probs_checker):
    
    neg = call_func(q)
    probs_checker = checker(q, res)
    
    return {
     'Callable': call_func, 'desc':desc, 'probs_checker':probs_checker, 'orig_statement': q, 'negated_statement':neg 
        
    }
    
    
    