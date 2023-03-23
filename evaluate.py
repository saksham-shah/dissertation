import torch
from data import *
import random
from utils.prepare_tensors import *
from utils.rpn import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def indexesFromTokens(lang, tokens):
#     return [lang.token2index[token] for token in tokens]

# def tensorFromTokens(lang, tokens):
#     indexes = indexesFromTokens(lang, tokens)
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# def tensorFromTokens(lang, tokens):
#     indexes = indexesFromTokens(lang, tokens)
#     tensor = torch.tensor(indexes, dtype=torch.long, device=device)
#     return tensor

def evaluate(config, embedding, encoder, decoder, tokens, numbers, q_lang, a_lang, max_length = 120):
    with torch.no_grad():
        input_tensor = tensorFromTokens(q_lang, tokens).view(-1, 1)
        input_length = input_tensor.size()[0]
        
        input_tensor = embedding(input_tensor, numbers, torch.tensor([0], device=device))
        encoder_outputs, encoder_hidden = encoder(input_tensor, [input_length], torch.tensor([0], device=device))

        decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        decoder_hidden = (encoder_hidden[0][:decoder.num_layers], encoder_hidden[1][:decoder.num_layers])

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            if config["attention"]:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di, :decoder_attention.shape[1]] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(a_lang.index2token[topi.item()])

            decoder_input = topi.squeeze().detach().unsqueeze(0)

        return decoded_words, decoder_attentions[:di + 1]

# def evaluateRandomly(embedding, mwps, encoder, decoder, q_lang, a_lang, n=10):
#     for i in range(n):
#         mwp = random.choice(mwps)
#         print('>', ' '.join(mwp.q_tokens))
#         print('=', ' '.join(mwp.a_tokens))
#         output_words, attentions = evaluate(embedding, encoder, decoder, mwp.q_tokens, q_lang, a_lang)
#         output_sentence = ' '.join(output_words)
#         print('<', output_sentence)
#         print('')

def check(config, output_tokens, target_tokens):
    if config["rpn"]:
        target_tokens = infix_to_rpn(target_tokens)
    output_sentence = ' '.join(output_tokens)
    target_sentence = ' '.join(target_tokens) + ' <EOS>'
    return output_sentence == target_sentence

def accuracy(config, mwps, embedding, encoder, decoder, q_lang, a_lang):
    correct = 0
    for mwp in mwps:
        q_tokens, a_tokens, numbers = tokensFromMWP(mwp.question, mwp.equation)
        q_tokens, a_tokens = mwp.question.split(" "), mwp.equation.split(" ")
        numbers = list(map(float, mwp.numbers.split(",")))
        output_words, attentions = evaluate(config, embedding, encoder, decoder, q_tokens, [numbers], q_lang, a_lang)

        if check(config, output_words, a_tokens):
            correct += 1

        if config["rpn"]:
            a_tokens = infix_to_rpn(a_tokens)

        output_sentence = ' '.join(output_words)
        target_sentence = ' '.join(a_tokens) + ' <EOS>'
        print(target_sentence, output_sentence)
    print("Accuracy:", correct / len(mwps))
    return correct / len(mwps)
