# Returns True if c is an operator (+-*/)
def is_operator(c):
    return c in "+-*/"

# Returns True if the precedence of operator x is less than or equal to the precedence of operator y
# e.g. * has higher precedence than +
def is_leq_precedence(x, y):
    return (x in "+-") or (y in "*/")

def infix_to_rpn(tokens):
    
    output = []
    operators = []
    
    for token in tokens:
        if is_operator(token):
            while len(operators) > 0 and is_operator(operators[-1]) and is_leq_precedence(token, operators[-1]):
                output.append(operators.pop())
            operators.append(token)
        elif token == '(':
            operators.append(token)
        elif token ==')':
            while len(operators) > 0 and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
        else:
            output.append(token)
    
    while len(operators) > 0:
        output.append(operators.pop())

    return output