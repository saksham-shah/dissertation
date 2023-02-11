import torch
import math
import random
import time
from models.embedding import Embedding
from models.encoder import Encoder
from models.decoder import Decoder
from utils.prepare_tensors import *
from config import *

# num_layers = 1

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Build_Data(Dataset):
    def __init__(self):
        # self.x = [mwp.q_tokens for mwp in valid_mwps]
        # self.y = [mwp.a_tokens for mwp in valid_mwps]
        self.len = len(valid_mwps)

    def __getitem__(self, index):
        return {'question': valid_mwps[index].full_question, 'formula': valid_mwps[index].target, 'answer': valid_mwps[index].answer}

    def __len__(self):
        return self.len

# BATCH_SIZE = 8

def batch_data(config):
    dataset = Build_Data()
    train_loader = DataLoader(dataset, batch_size=config["batch_size"])
    return train_loader

# print(train_loader)

# teacher_forcing_ratio = 0.9

def train(config, input_tensor, target_tensor, input_lengths, target_lengths, embedding, encoder, decoder, embedding_optimiser, encoder_optimiser, decoder_optimiser, criterion):
    batch_size = input_tensor.shape[1]

    embedding_optimiser.zero_grad()
    encoder_optimiser.zero_grad()
    decoder_optimiser.zero_grad()

    sorted_input, sorted_lengths, restore_indexes = sort_by_length(input_tensor, input_lengths)
    sorted_input = embedding(sorted_input)
    encoder_outputs, encoder_hidden = encoder(sorted_input, sorted_lengths, restore_indexes)

    loss = 0
    
    decoder_input = torch.tensor([SOS_token for i in range(batch_size)], device=device)
    decoder_hidden = (encoder_hidden[0][:encoder.num_layers], encoder_hidden[1][:encoder.num_layers])

    target_length = max(target_lengths)
    use_teacher_forcing = True if random.random() < config["teacher_forcing_ratio"] else False

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

    embedding_optimiser.step()
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

def trainIters(config, embedding, encoder, decoder, n_iters, print_every=1000):
    start = time.time()
    print_loss_total = 0

    lr = config["learning_rate"]

    embedding_optimiser = torch.optim.Adam(embedding.parameters(), lr=lr)
    encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimiser = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()

    count = 0

    train_loader = batch_data(config)

    for iter in range(1, n_iters + 1):
        for mwp in train_loader:
            input_tensor, target_tensor, input_lengths, target_lengths, numbers = indexesFromPairs(mwp['question'], mwp['formula'])
            count += 1

            loss = train(config, input_tensor, target_tensor, input_lengths, target_lengths, embedding, encoder, decoder, embedding_optimiser, encoder_optimiser, decoder_optimiser, criterion)
            print_loss_total += loss

            if count % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, count / n_iters / len(train_loader)), count, count / n_iters / len(train_loader) * 100, print_loss_avg))

# embedding_size = 300
# hidden_size = 256
embedding = Embedding(config, q_lang.n_tokens, EMBEDDING_SIZE, q_weights_matrix).to(device)
encoder = Encoder(config, EMBEDDING_SIZE).to(device)
attn_decoder = Decoder(config, a_lang.n_tokens, EMBEDDING_SIZE).to(device)

trainIters(config, embedding, encoder, attn_decoder, 1, print_every=100)

torch.save(embedding, 'asdiv-baseline-embedding.pt')
torch.save(encoder, 'asdiv-baseline-encoder.pt')
torch.save(attn_decoder, 'asdiv-baseline-attndecoder.pt')