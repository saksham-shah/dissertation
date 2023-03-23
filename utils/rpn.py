# Returns True if c is an operator (+-*/)
def is_operator(t):
    return t in "+-*/"

# Returns True if the precedence of operator x is less than or equal to the precedence of operator y
# e.g. * has higher precedence than +
def is_leq_precedence(x, y):
    return (x in "+-") or (y in "*/")

def do_operation(op, num1, num2):
    if op == '+':
        return num1 + num2
    if op == '-':
        return num1 - num2
    if op == '*':
        return num1 * num2
    if op == '/':
        return num1 / num2

def getIndex(t):
    if t[0] != '#':
        return None
    try:
        return int(t[1:])
    except ValueError:
        return None

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
            if len(operators) == 0:
                return None
            operators.pop()
        else:
            output.append(token)
    
    while len(operators) > 0:
        output.append(operators.pop())

    return output

def eval_rpn(tokens, numbers):
    if tokens is None:
        return None
    
    stack = []
    for token in tokens:
        if is_operator(token):
            if len(stack) < 2:
                return None
            num2 = stack.pop()
            num1 = stack.pop()
            if token == '/' and num2 == 0.0:
                return None
            stack.append(do_operation(token, num1, num2))
        else:
            index = getIndex(token)
            if index is None or index >= len(numbers):
                return None
            stack.append(numbers[index])
    
    if len(stack) != 1:
        return None
    
    return stack[0]