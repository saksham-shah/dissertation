from utils.process_input import *

# Process input: string_to_float

string_to_float_tests = [
    ("3", 3.0),
    ("four", 4.0),
    ("forty-five", 45.0),
    ("1,232", 1232.0),
    ("house", None),
]

for inp, out in string_to_float_tests:
    assert string_to_float(inp) == out

# Process input: tokenise_question

tokenise_question_tests = [
    ("Hello", ["hello"]),
    ("Hello world !", ["hello", "world", "!"]),
    ("Hello     world    !", ["hello", "world", "!"]),
]

for inp, out in tokenise_question_tests:
    assert tokenise_question(inp) == out

# Process input: tokenise_formula

tokenise_formula_tests = [
    ("3+4", ["3", "+", "4"]),
    ("(3+4)     -5", ["(", "3", "+", "4", ")", "-", "5"]),
    ("4.5/2", ["4.5", "/", "2"]),
]

for inp, out in tokenise_formula_tests:
    assert tokenise_formula(inp) == out

# Process input: tokens_from_MWP

tokens_from_MWP_tests = [
    ("3 plus 4", "3+4", ["#0", "plus", "#1"], ["#0", "+", "#1"], [3, 4]),
    ("3 and 4 also 5", "3/5", ["#0", "and", "#1", "also", "#2"], ["#0", "/", "#2"], [3, 4, 5]),
    ("3 dozen", "3*12", None, None, None),
]

for question, equation, q_tokens, a_tokens, nums in tokens_from_MWP_tests:
    res_q_tokens, res_a_tokens, res_nums = tokens_from_MWP(question, equation)
    assert res_q_tokens == q_tokens and res_a_tokens == a_tokens and res_nums == nums

print("All tests passed.")