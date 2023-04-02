import re
from word2number import w2n
from utils.rpn import *

knowledge = {
    "dozen": 12.0,
    "cents": 0.01,
    "dimes": 0.1,
    "%": 0.01,
    "percent": 0.01
}

def string_to_float(s):
    try:
        return float(s)
    except ValueError:
        try:
            return float(w2n.word_to_num(s))
        except ValueError:
            if "," in s and len(s) > 1:
                return string_to_float(re.sub(r",", r"", s))
            # if s in knowledge:
            #     return knowledge[s]
            return None

def tokenise_question(s):
    s = s.lower()
    s = re.sub(r" +", r" ", s)
    return s.strip().split(" ")

def tokenise_formula(s):
    s = re.sub(r"([()+\-*/=])", r" \1 ", s)
    s = re.sub(r" +", r" ", s)
    return s.strip().split(" ")

def tokensFromMWP(question, formula):
    q_tokens = tokenise_question(question)
    a_tokens = tokenise_formula(formula)

    numbers = []
    
    for i in range(len(q_tokens)):
        num = string_to_float(q_tokens[i])
        if num is not None:
            # if num not in numbers:
            numbers.append(num)
            q_tokens[i] = "#" + str(len(numbers) - 1)
            # else:
            #     print(q_tokens)
    
    for i in range(len(a_tokens)):
        num = string_to_float(a_tokens[i])
        if num is not None:
            if num in numbers:
                a_tokens[i] = "#" + str(numbers.index(num))
            else:
                # print(question, formula)
                # Drawbacks: real world knowledge, e.g. money, 'dozen'; numbers as lists of people; percentages; errors in dataset, non-trivial calculations
                return None, None, None
    
    # if rpn:
    #     a_tokens = infix_to_rpn(a_tokens)
    
    return q_tokens, a_tokens, numbers