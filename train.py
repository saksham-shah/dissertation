import torch
import math
import random
import time
from utils.prepare_tensors import *
from utils.load_batches import *
from evaluate import *

def train(config, input_tensor, target_tensor, input_lengths, target_lengths, numbers, embedding, encoder, decoder, embedding_optimiser, encoder_optimiser, decoder_optimiser, criterion):
    batch_size = input_tensor.shape[1]

    embedding_optimiser.zero_grad()
    encoder_optimiser.zero_grad()
    decoder_optimiser.zero_grad()

    sorted_input, sorted_lengths, restore_indexes = sort_by_length(input_tensor, input_lengths)
    sorted_input = embedding(sorted_input, numbers)
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

# def validate

def trainIters(config, embedding, encoder, decoder, n_iters, print_every=1000):
    start = time.time()
    print_loss_total = 0

    lr = config["learning_rate"]

    embedding_optimiser = torch.optim.Adam(embedding.parameters(), lr=lr)
    encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimiser = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss()

    count = 0

    max_acc = 0
    epoch_since_improvement = 0

    train_loader, test = train_test(config, valid_mwps)

    for iter in range(1, n_iters + 1):
        for mwp in train_loader:
            input_tensor, target_tensor, input_lengths, target_lengths, numbers = indexesFromPairs(mwp['question'], mwp['formula'], config["rpn"])
            count += 1

            loss = train(config, input_tensor, target_tensor, input_lengths, target_lengths, numbers, embedding, encoder, decoder, embedding_optimiser, encoder_optimiser, decoder_optimiser, criterion)
            print_loss_total += loss

            if count % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, count / n_iters / len(train_loader)), count, count / n_iters / len(train_loader) * 100, print_loss_avg))
        
        correct = 0
        for mwp in test:
            q_tokens, a_tokens, numbers = tokensFromMWP(mwp.full_question, mwp.target)
            output_words, attentions = evaluate(embedding, encoder, decoder, q_tokens, numbers)
            if check(config, output_words, a_tokens):
                correct += 1
            
        acc = correct / len(test)
        print ("%s epoch: %d, accurary: %.4f" % (timeSince(start, count / n_iters / len(train_loader)), iter, acc))

        if acc > max_acc:
            max_acc = acc
            epoch_since_improvement = 0
        else:
            epoch_since_improvement += 1
        
        if config["early_stopping"] >= 0 and epoch_since_improvement > config["early_stopping"]:
            print("Early stopping at Epoch %d, after no improvement in %d epochs" % (iter, epoch_since_improvement))
            break
    
    return max_acc, acc, iter

