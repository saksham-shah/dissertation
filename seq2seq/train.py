import torch
import math
import random
import time
from utils.prepare_tensors import *
from utils.load_batches import *
from seq2seq.evaluate import *

# Train model on single input
def train(config, input_tensor, target_tensor, input_lengths, target_lengths, numbers, embedding, encoder, decoder, embedding_optimiser, encoder_optimiser, decoder_optimiser, criterion):
    batch_size = input_tensor.shape[1]

    embedding_optimiser.zero_grad()
    encoder_optimiser.zero_grad()
    decoder_optimiser.zero_grad()

    # Sort inputs to use pack_padded_sequences in Encoder
    sorted_input, sorted_lengths, restore_indexes = sort_by_length(input_tensor, input_lengths)

    # Encode into token embeddings and pass into encoder
    sorted_input = embedding(sorted_input, numbers, restore_indexes)
    encoder_outputs, encoder_hidden = encoder(sorted_input, sorted_lengths, restore_indexes)

    loss = 0
    
    # Initialise decoder input and internal hidden state
    decoder_input = torch.tensor([SOS_token for i in range(batch_size)], device=device)
    decoder_hidden = (encoder_hidden[0][:encoder.num_layers], encoder_hidden[1][:encoder.num_layers])

    target_length = max(target_lengths)
    use_teacher_forcing = True if random.random() < config["teacher_forcing_ratio"] else False

    if use_teacher_forcing: # feed model correct output at each step
        for di in range(target_length):
            if config["attention"]:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di]) # compute total loss
            decoder_input = target_tensor[di]
    else: # feed model its own output at each step
        for di in range(target_length):
            if config["attention"]:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])
    
    loss.backward()

    embedding_optimiser.step()
    encoder_optimiser.step()
    decoder_optimiser.step()

    return loss.item() / target_length

# Time utils for printing progress
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

# Train model on train set for given number of epochs (n_iters)
def trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, n_iters, print_every=1000):
    start = time.time()
    print_loss_total = 0

    lr = config["learning_rate"]

    embedding_optimiser = torch.optim.Adam(embedding.parameters(), lr=lr)
    encoder_optimiser = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimiser = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = torch.nn.NLLLoss() # negative log likelihood for finite output vocabulary

    count = 0

    max_acc = 0
    acc = 0
    iter = 0
    epoch_since_improvement = 0
    print_loss_avg = 0

    for iter in range(1, n_iters + 1):
        for mwp in train_loader:
            # Prepare input and target tensors
            input_tensor, target_tensor, input_lengths, target_lengths, numbers = indexes_from_pairs(mwp['question'], mwp['formula'], q_lang, a_lang, config["rpn"])
            count += 1

            numbers = [list(map(float, nums.split(","))) for nums in mwp['numbers']]

            # Train model on example
            loss = train(config, input_tensor, target_tensor, input_lengths, target_lengths, numbers, embedding, encoder, decoder, embedding_optimiser, encoder_optimiser, decoder_optimiser, criterion)
            print_loss_total += loss

            if print_every > 0 and count % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, count / n_iters / len(train_loader)), count, count / n_iters / len(train_loader) * 100, print_loss_avg))
        
        if print_every == 0:
            print_loss_avg = print_loss_total / len(train_loader)
            print_loss_total = 0

        # Compute model accuracy
        acc = accuracy(config, test_loader, embedding, encoder, decoder, q_lang, a_lang)
        print ("%s epoch: %d, accurary: %.4f, loss: %.4f" % (timeSince(start, count / n_iters / len(train_loader)), iter, acc, print_loss_avg))

        # Early stopping if no improvement in given number of epochs
        if acc > max_acc:
            max_acc = acc
            epoch_since_improvement = 0
        else:
            epoch_since_improvement += 1
        
        if config["early_stopping"] >= 0 and epoch_since_improvement > config["early_stopping"]:
            print("Early stopping at Epoch %d, after no improvement in %d epochs" % (iter, epoch_since_improvement))
            break
    
    return max_acc, acc, iter

