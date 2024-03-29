import torch
import json
import re
from utils.process_input import *
from seq2seq.evaluate import *
from models import *

config = {
    "num_layers": 1,
    "batch_size": 1,
    "teacher_forcing_ratio": 0.9,
    "learning_rate": 0.0005,
    "hidden_size": 256,
    "bidirectional": True,
    "dropout": 0.1,
    "early_stopping": 5,
    "rpn": False,
    "num_emb": True,
    "embedding_size": 300,
    "dataset": ["asdiv", "mawps"],
    "attention": True,
}

# Load model and vocabularies from .pt file
def load_model(file):
    print("Loading...")
    model = torch.load(file, map_location=torch.device('cpu'))
    return model['embedding'], model['encoder'], model['decoder'], model['token2index'], model['index2token']

embedding, encoder, decoder, token2index, index2token = load_model('model/rpn_both.pt')

while True:
    # Take user input
    question = input("Question: ")

    # Tokenise equation
    question = re.sub(r"([.,])([^0-9])", r" \1 \2", question)
    tokens = tokenise_question(question)

    # Replace numbers with abstract tokens
    numbers = []
    for i in range(len(tokens)):
        num = string_to_float(tokens[i])
        if num is not None:
            tokens[i] = "#" + str(len(numbers))
            numbers.append(num)
    
    # Generate model output
    output_tokens, attentions = evaluate(config, embedding, encoder, decoder, tokens, [numbers], token2index, index2token)

    # Evaluate output to get answer
    if output_tokens[-1] == '<EOS>':
        output_tokens.pop()    
        if config['rpn']:
            eval_output_tokens = output_tokens
        else:
            eval_output_tokens = infix_to_rpn(output_tokens)
        output_ans = eval_rpn(eval_output_tokens, numbers)

    # Substitute numbers back into output
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