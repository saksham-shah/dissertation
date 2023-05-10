import torch
from data.load_data import *
from utils.process_input import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert tokens into indexes from vocabulary
def indexes_from_tokens(token2index, tokens):
    def index_from_token(token2index, token):
        if token in token2index:
            return token2index[token]
        return token2index['UNK'] # use UNK for unseen tokens
    
    indexes = [index_from_token(token2index, token) for token in tokens]
    indexes.append(EOS_token)
    return indexes

# Convert tokens into tensors
def tensor_from_tokens(token2index, tokens):
    indexes = indexes_from_tokens(token2index, tokens)
    tensor = torch.tensor(indexes, dtype=torch.long, device=device)
    return tensor

# Pad all indexes in batch to length of longest sequence
def pad_indexes(indexes, max_length):
    indexes += [EOS_token for i in range(max_length - len(indexes))]
    return indexes

# Convert input and target to tensors for model
def indexes_from_pairs(questions, formulas, q_lang, a_lang, rpn=False):
    # Get input and target indexes
    q_indexes = [indexes_from_tokens(q_lang.token2index, q.split(" ")) for q in questions]
    a_indexes = [indexes_from_tokens(a_lang.token2index, a.split(" ")) for a in formulas]

    numbers = []

    q_lengths = [len(q) for q in q_indexes]
    a_lengths = [len(a) for a in a_indexes]

    # Pad sequences to equal length
    q_padded = [pad_indexes(q, max(q_lengths)) for q in q_indexes]
    a_padded = [pad_indexes(a, max(a_lengths)) for a in a_indexes]

    # Convert sequences to tensor
    q_tensor = torch.tensor(q_padded, device=device).transpose(0, 1)
    a_tensor = torch.tensor(a_padded, device=device).transpose(0, 1)

    return q_tensor, a_tensor, q_lengths, a_lengths, numbers

# Sort sequences by length
# Needed for pack_padded_sequence
def sort_by_length(sequences, lengths):
    restore_indexes = list(range(sequences.size(1)))

    sorted_indexes = torch.tensor(sorted(restore_indexes, key=lambda x: lengths[x], reverse=True), device=device)
    sorted_sequences = sequences.index_select(1, sorted_indexes)
    sorted_lengths = [lengths[i] for i in sorted_indexes]

    restore_indexes = torch.tensor(sorted(restore_indexes, key=lambda x: sorted_indexes[x]), device=device)

    return sorted_sequences, sorted_lengths, restore_indexes
    