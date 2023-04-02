import torch
import json
import re
from utils.process_input import *
from evaluate import *
from config import *
def load_model(path='model/'):
    print("Loading...")
    encoder = torch.load(path + 'encoder.pt')
    decoder = torch.load(path + 'decoder.pt')
    embedding = torch.load(path + 'embedding.pt')

    with open(path + 'token2index.json') as file:
        token2index = json.loads(file.read())

    with open(path + 'index2token.json') as file:
        index2token_str = json.loads(file.read())
      
    index2token = {}
    for key in index2token_str:
      index2token[int(key)] = index2token_str[key]

    print("Loaded.")
    return embedding, encoder, decoder, token2index, index2token

embedding, encoder, decoder, token2index, index2token = load_model()

while True:
    question = input("Question:")
    question = re.sub(r"([.,])([^0-9])", r" \1 \2", question)
    tokens = tokenise_question(question)
    numbers = []
    for i in range(len(tokens)):
        num = string_to_float(tokens[i])
        if num is not None:
            tokens[i] = "#" + str(len(numbers))
            numbers.append(num)
    
    output_tokens, attentions = evaluate(config, embedding, encoder, decoder, tokens, [numbers], token2index, index2token)
    
    if output_tokens[-1] == '<EOS>':
        output_tokens.pop()    
        if not config['rpn']:
            output_tokens = infix_to_rpn(output_tokens)
        output_ans = eval_rpn(output_tokens, numbers)

    for i in range(len(output_tokens)):
        if output_tokens[i][0] == '#':
            index = int(output_tokens[i][1:])
            if index < len(numbers):
                output_tokens[i] = str(numbers[index])
            else:
                output_tokens[i] = '?'

    if output_ans is None:
      output_ans = '?'
    else:
      output_ans = str(output_ans)

    print(" ".join(output_tokens) + " = " + output_ans)