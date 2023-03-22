import torch
from data import *
from utils.load_batches import *
from evaluate import *
from config import *
from timeit import default_timer as timer


from models.transformer import Transformer

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == EOS_token).transpose(0, 1)
    tgt_padding_mask = (tgt == EOS_token).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

mwps, q_lang, a_lang = load_data(config)

transformer = Transformer(config, 5, 5, 512, 8, q_lang.n_tokens, a_lang.n_tokens)

optimiser = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'])
criterion = torch.nn.CrossEntropyLoss(ignore_index=EOS_token)

def train(config, model, train_loader, optimiser, criterion):
    model.train()

    losses = 0

    for mwp in train_loader:
        input_tensor, target_tensor, input_lengths, target_lengths, numbers = indexesFromPairs(mwp['question'], mwp['formula'], config["rpn"])
        src, tgt = input_tensor, target_tensor

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimiser.zero_grad()

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimiser.step()
        losses += loss.item()
    return losses / len(list(train_loader))

def evaluate(config, model, val_loader, criterion):
    model.eval()
    losses = 0

    for mwp in val_loader:
        input_tensor, target_tensor, input_lengths, target_lengths, numbers = indexesFromPairs(mwp['question'], mwp['formula'], config["rpn"])
        src, tgt = input_tensor, target_tensor

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(list(val_loader))
        

NUM_EPOCHS = 18

train_loader, val_loader = train_test(config, mwps, batch_test=True)

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train(config, transformer, train_loader, optimiser, criterion)
    end_time = timer()
    val_loss = evaluate(config, transformer, val_loader, criterion)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))