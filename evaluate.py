import torch
from data import *
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromTokens(lang, tokens):
    return [lang.token2index[token] for token in tokens]

def tensorFromTokens(lang, tokens):
    indexes = indexesFromTokens(lang, tokens)
    indexes.append(EOS_token)
    # while len(indexes) < MAX_LENGTH:
    #     indexes.append(PAD_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluate(encoder, decoder, tokens, max_length = 120):
    with torch.no_grad():
        input_tensor = tensorFromTokens(q_lang, tokens)
        input_length = input_tensor.size()[0]
        # encoder_hidden = None # encoder.init_hidden()

        # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # for ei in range(input_length):
        #     encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        #     encoder_outputs[ei] += encoder_output[0, 0]
        
        encoder_outputs, encoder_hidden = encoder(input_tensor, [input_length], torch.LongTensor([0], device=device))

        decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        # decoder_hidden = encoder_hidden
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

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        mwp = random.choice(valid_mwps)
        print('>', ' '.join(mwp.q_tokens))
        print('=', ' '.join(mwp.a_tokens))
        output_words, attentions = evaluate(encoder, decoder, mwp.q_tokens)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def accuracy(encoder, decoder):
    correct = 0
    for mwp in valid_mwps:
        output_words, attentions = evaluate(encoder, decoder, mwp.q_tokens)
        output_sentence = ' '.join(output_words)
        target_sentence = ' '.join(mwp.a_tokens) + ' <EOS>'
        if output_sentence == target_sentence:
            correct += 1
        print(target_sentence, output_sentence)
    print("Accuracy:", correct / len(valid_mwps))

encoder = torch.load('asdiv-baseline-encoder.pt', map_location=device)
attn_decoder = torch.load('asdiv-baseline-attndecoder.pt', map_location=device)

# evaluateRandomly(encoder, attn_decoder)
accuracy(encoder, attn_decoder)