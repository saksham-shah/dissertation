import torch
from data.load_data import *
from utils.process_input import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromTokens(token2index, tokens):
    def index_from_token(token2index, token):
        if token in token2index:
            return token2index[token]
        return token2index['UNK']
    
    indexes = [index_from_token(token2index, token) for token in tokens]
    indexes.append(EOS_token)
    return indexes

def tensorFromTokens(token2index, tokens):
    indexes = indexesFromTokens(token2index, tokens)
    tensor = torch.tensor(indexes, dtype=torch.long, device=device)
    return tensor

# def tensorsFromPair(lang, pair):
#     input_tensor = tensorFromTokens(token2index, pair[0])
#     target_tensor = tensorFromTokens(token2index, pair[1])
#     return (input_tensor, target_tensor)

def pad_indexes(indexes, max_length):
    indexes += [EOS_token for i in range(max_length - len(indexes))]
    return indexes

def indexesFromPairs(questions, formulas, q_lang, a_lang, rpn=False):
    # all_tokens = [tokensFromMWP(question, formula, rpn=rpn) for question, formula in zip(questions, formulas)]
    # q_indexes = [indexesFromTokens(q_lang, q) for q,_,_ in all_tokens]
    # a_indexes = [indexesFromTokens(a_lang, a) for _,a,_ in all_tokens]
    # numbers = [n for _,_,n in all_tokens]

    q_indexes = [indexesFromTokens(q_lang.token2index, q.split(" ")) for q in questions]
    a_indexes = [indexesFromTokens(a_lang.token2index, a.split(" ")) for a in formulas]

    numbers = []

    q_lengths = [len(q) for q in q_indexes]
    a_lengths = [len(a) for a in a_indexes]

    q_padded = [pad_indexes(q, max(q_lengths)) for q in q_indexes]
    a_padded = [pad_indexes(a, max(a_lengths)) for a in a_indexes]

    q_tensor = torch.tensor(q_padded, device=device).transpose(0, 1)
    a_tensor = torch.tensor(a_padded, device=device).transpose(0, 1)

    return q_tensor, a_tensor, q_lengths, a_lengths, numbers

def sort_by_length(sequences, lengths):
    restore_indexes = list(range(sequences.size(1)))

    sorted_indexes = torch.tensor(sorted(restore_indexes, key=lambda x: lengths[x], reverse=True), device=device)
    sorted_sequences = sequences.index_select(1, sorted_indexes)
    sorted_lengths = [lengths[i] for i in sorted_indexes]

    restore_indexes = torch.tensor(sorted(restore_indexes, key=lambda x: sorted_indexes[x]), device=device)

    return sorted_sequences, sorted_lengths, restore_indexes
    