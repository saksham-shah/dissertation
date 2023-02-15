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

def evaluate(embedding, encoder, decoder, tokens, max_length = 120):
    with torch.no_grad():
        input_tensor = tensorFromTokens(q_lang, tokens).view(-1, 1)
        input_length = input_tensor.size()[0]
        
        input_tensor = embedding(input_tensor)
        encoder_outputs, encoder_hidden = encoder(input_tensor, [input_length], torch.LongTensor([0], device=device))

        decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        decoder_hidden = (encoder_hidden[0][:decoder.num_layers], encoder_hidden[1][:decoder.num_layers])

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di, :decoder_attention.shape[1]] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(a_lang.index2token[topi.item()])

            decoder_input = topi.squeeze().detach().unsqueeze(0)

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(embedding, encoder, decoder, n=10):
    for i in range(n):
        mwp = random.choice(valid_mwps)
        print('>', ' '.join(mwp.q_tokens))
        print('=', ' '.join(mwp.a_tokens))
        output_words, attentions = evaluate(embedding, encoder, decoder, mwp.q_tokens)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def check(config, output_tokens, target_tokens):
    if config["rpn"]:
        target_tokens = infix_to_rpn(target_tokens)
    output_sentence = ' '.join(output_tokens)
    target_sentence = ' '.join(target_tokens) + ' <EOS>'
    return output_sentence == target_sentence

def accuracy(config, embedding, encoder, decoder):
    correct = 0
    for mwp in valid_mwps:
        q_tokens, a_tokens, _ = tokensFromMWP(mwp.full_question, mwp.target)
        output_words, attentions = evaluate(embedding, encoder, decoder, q_tokens)

        if check(config, output_words, a_tokens):
            correct += 1

        output_sentence = ' '.join(output_words)
        target_sentence = ' '.join(a_tokens) + ' <EOS>'
        print(target_sentence, output_sentence)
    print("Accuracy:", correct / len(valid_mwps))
    return correct / len(valid_mwps)
