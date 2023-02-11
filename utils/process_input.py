import re
from word2number import w2n

def string_to_float(s):
    try:
        return float(s)
    except ValueError:
        try:
            return float(w2n.word_to_num(s))
        except ValueError:
            if "," in s and len(s) > 1:
                return string_to_float(re.sub(r",", r"", s))
            return None

def tokenise_question(s):
    s = s.lower()
    s = re.sub(r"([?$])", r" \1 ", s)
    s = re.sub(r"([.,])([^0-9])", r" \1 \2", s)
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
            if num not in numbers:
                numbers.append(num)
                q_tokens[i] = "#" + str(len(numbers) - 1)
            else:
                print(q_tokens)
    
    for i in range(len(a_tokens)):
        num = string_to_float(a_tokens[i])
        if num is not None and num in numbers:
            a_tokens[i] = "#" + str(numbers.index(num))
    
    return q_tokens, a_tokens, numbers