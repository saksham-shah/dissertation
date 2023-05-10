from utils.rpn import *

# RPN: infix_to_rpn

def test_infix_to_rpn(infix, postfix):
    infix = infix.split(" ")
    result = infix_to_rpn(infix)
    result = " ".join(result)
    assert result == postfix

infix_to_rpn_tests = [
    ("#0 + #1", "#0 #1 +"),
    ("( #0 - #2 ) + #1", "#0 #2 - #1 +"),
    ("#0 - ( #2 + #1 )", "#0 #2 #1 + -"),
    ("#0 - #2 * #1", "#0 #2 #1 * -"),
    ("#0 * #2 - #1", "#0 #2 * #1 -"),
]

for test in infix_to_rpn_tests:
    test_infix_to_rpn(*test)

# RPN: eval_rpn

def test_eval_rpn(rpn, numbers, answer):
    rpn = rpn.split(" ")
    result = eval_rpn(rpn, numbers)
    assert result == answer

numbers = [1, 2, 3]
eval_rpn_tests = [
    ("#0 #1 +", 3),
    ("#0 #2 - #1 +", 0),
    ("#0 #2 #1 + -", -4),
    ("#0 #2 #1 * -", -5),
    ("#0 #2 * #1 -", 1),
]

for rpn_expr, answer in eval_rpn_tests:
    test_eval_rpn(rpn_expr, numbers, answer)

print("All tests passed.")