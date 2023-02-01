import torch
import math
import random
import time
from models.embedding import Embedding
from models.encoder import Encoder
from models.decoder import Decoder
from utils import *

num_layers = 1

# def indexesFromTokens(lang, tokens):
#     return [lang.token2index[token] for token in tokens]


# def tensorFromTokens(lang, tokens):
#     indexes = indexesFromTokens(lang, tokens)
#     indexes.append(EOS_token)
#     # while len(indexes) < MAX_LENGTH:
#     #     indexes.append(PAD_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device)

# def tensorsFromPair(lang, pair):
#     input_tensor = tensorFromTokens(lang, pair[0])
#     target_tensor = tensorFromTokens(lang, pair[1])
#     return (input_tensor, target_tensor)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Build_Data(Dataset):
    def __init__(self):
        self.x = [mwp.q_tokens for mwp in valid_mwps]
        self.y = [mwp.a_tokens for mwp in valid_mwps]
        self.len = len(valid_mwps)

    def __getitem__(self, index):
        return {'question': valid_mwps[index].full_question, 'formula': valid_mwps[index].target, 'answer': valid_mwps[index].answer}
        input_tensor = tensorFromTokens(q_lang, self.x[index])
        target_tensor = tensorFromTokens(a_lang, self.y[index])
        return input_tensor, target_tensor

    def __len__(self):
        return self.len

BATCH_SIZE = 8

dataset = Build_Data()
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

print(train_loader)

teacher_forcing_ratio = 0.9

def train(input_tensor, target_tensor, input_lengths, target_lengths, encoder, decoder, encoder_optimiser, decoder_optimiser, criterion):
    batch_size = input_tensor.shape[1]
    # encoder_hidden = None # encoder.init_hidden(batch_size=batch_size)

    encoder_optimiser.zero_grad()
    decoder_optimiser.zero_grad()

    sorted_input, sorted_lengths, restore_indexes = sort_by_length(input_tensor, input_lengths)
    encoder_outputs, encoder_hidden = encoder(sorted_input, sorted_lengths, restore_indexes)

    loss = 0
    
    decoder_input = torch.tensor([SOS_token for i in range(batch_size)], device=device)
    decoder_hidden = (encoder_hidden[0][:num_layers], encoder_hidden[1][:num_layers])

    target_length = max(target_lengths)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
            # if decoder_input.item() == EOS_token:
            #     break
    
    loss.backward()

    encoder_optimiser.step()
    decoder_optimiser.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, learning_rate=0.0005):
    start = time.time()
    print_loss_total = 0

    encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=learning_rate) #adam
    decoder_optimiser = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    count = 0

    for iter in range(1, n_iters + 1):
        for mwp in train_loader:
            input_tensor, target_tensor, input_lengths, target_lengths, numbers = indexesFromPairs(mwp['question'], mwp['formula'])
            count += 1
            # input_tensor = torch.transpose(input_tensor, 0, 1)
            # target_tensor = torch.transpose(target_tensor, 0, 1)
            loss = train(input_tensor, target_tensor, input_lengths, target_lengths, encoder, decoder, encoder_optimiser, decoder_optimiser, criterion)
            print_loss_total += loss

            if count % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, count / n_iters / len(train_loader)), count, count / n_iters / len(train_loader) * 100, print_loss_avg))

# embedding_size = 300
hidden_size = 256
encoder = Encoder(q_lang.n_tokens, EMBEDDING_SIZE, hidden_size, num_layers=num_layers).to(device)
attn_decoder = Decoder(a_lang.n_tokens, EMBEDDING_SIZE, hidden_size, num_layers=num_layers).to(device)

trainIters(encoder, attn_decoder, 50, print_every=100)

torch.save(encoder, 'asdiv-baseline-encoder.pt')
torch.save(attn_decoder, 'asdiv-baseline-attndecoder.pt')