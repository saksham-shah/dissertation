import re
from word2number import w2n
from utils.rpn import *

# Convert string to float using word2number
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

# Remove excess whitespace and split by space
def tokenise_question(s):
    s = s.lower()
    s = re.sub(r" +", r" ", s)
    return s.strip().split(" ")

# Spaces between all characters and split by space
def tokenise_formula(s):
    s = re.sub(r"([()+\-*/=])", r" \1 ", s)
    s = re.sub(r" +", r" ", s)
    return s.strip().split(" ")

# Convert MWP to tokens
def tokens_from_MWP(question, formula):
    q_tokens = tokenise_question(question)
    a_tokens = tokenise_formula(formula)

    numbers = []
    
    # Replace numbers in question with abstract tokens
    for i in range(len(q_tokens)):
        num = string_to_float(q_tokens[i])
        if num is not None:
            numbers.append(num)
            q_tokens[i] = "#" + str(len(numbers) - 1)
    
    # Replace numbers in equation with same tokens as question
    for i in range(len(a_tokens)):
        num = string_to_float(a_tokens[i])
        if num is not None:
            if num in numbers:
                a_tokens[i] = "#" + str(numbers.index(num))
            else:
                # Invalid MWP as equation has unseen number
                # Could happen when: real world knowledge, e.g. money, 'dozen'; numbers as lists of people; percentages; errors in dataset, non-trivial calculations
                return None, None, None
    
    return q_tokens, a_tokens, numbers