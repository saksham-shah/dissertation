import torch
from data.load_data import *
import math
from utils.prepare_tensors import *
from utils.rpn import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate model predictions for single question
def evaluate(config, embedding, encoder, decoder, tokens, numbers, token2index, index2token, max_length = 120):
    with torch.no_grad():
        # Convert tokens to tensor
        input_tensor = tensor_from_tokens(token2index, tokens).view(-1, 1)
        input_length = input_tensor.size()[0]
        
        # Embed token embeddings and pass into encoder
        input_tensor = embedding(input_tensor, numbers, torch.tensor([0], device=device))
        encoder_outputs, encoder_hidden = encoder(input_tensor, [input_length], torch.tensor([0], device=device))

        # Initialise decoder input and hidden state
        decoder_input = torch.tensor([SOS_token], device=device)

        decoder_hidden = (encoder_hidden[0][:decoder.num_layers], encoder_hidden[1][:decoder.num_layers])

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        # Predict equation tokens
        for di in range(max_length):
            if config["attention"]:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di, :decoder_attention.shape[1]] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            
            # Greedy decoding - choose token with max probability
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token: # stop if EOS generated
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(index2token[topi.item()])

            decoder_input = topi.squeeze().detach().unsqueeze(0)

        return decoded_words, decoder_attentions[:di + 1]

# Check if model prediction is correct
def check(config, output_tokens, target_tokens, answer=None, numbers=None):
    if answer is not None and numbers is not None:
        if output_tokens[-1] == '<EOS>':
            output_tokens.pop()    
        if not config['rpn']: # equation should be in RPN to be evaluated
            output_tokens = infix_to_rpn(output_tokens)

        # Evaluate equation and get prediction answer
        output_ans = eval_rpn(output_tokens, numbers)
        # Mark as correct if result is sufficiently close to real answer
        return output_ans is not None and math.isclose(output_ans, answer, rel_tol=1e-4)

    # Fall-back accuracy check if answer/numbers not provided
    if config['rpn']:
        target_tokens = infix_to_rpn(target_tokens)

    output_sentence = ' '.join(output_tokens)
    target_sentence = ' '.join(target_tokens) + ' <EOS>'
    return output_sentence == target_sentence

# Evaluate and return model accuracy on test set
def accuracy(config, test_set, embedding, encoder, decoder, q_lang, a_lang, print_output=False):
    correct = 0
    for mwp in test_set:
        q_tokens, a_tokens = mwp['question'][0].split(" "), mwp['formula'][0].split(" ")
        numbers = list(map(float, mwp['numbers'][0].split(",")))

        # Get model output
        output_words, attentions = evaluate(config, embedding, encoder, decoder, q_tokens, [numbers], q_lang.token2index, a_lang.index2token)

        # Check if output is correct
        if check(config, output_words, a_tokens, mwp['answer'][0], numbers):
            correct += 1

        if print_output:
            if config["rpn"]:
                a_tokens = infix_to_rpn(a_tokens)

            output_sentence = ' '.join(output_words)
            target_sentence = ' '.join(a_tokens) + ' <EOS>'
            print(target_sentence, output_sentence)
    if print_output:
        print("Accuracy:", correct / len(test_set))
    return correct / len(test_set)