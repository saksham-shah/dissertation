import torch
from data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenise_question(s):
    s = s.lower()
    s = re.sub(r"([?$])", r" \1 ", s)
    s = re.sub(r"([.,])([^0-9])", r" \1 \2", s)
    s = re.sub(r" +", r" ", s)
    return s.strip().split(" ")

def tokenise_formula(s):
    s = re.sub(r"([()+\-*/=])", r" \1 ", s)
    s = re.sub(r" +", r" ", s)
    return s.strip().split(" ")

def tokensFromMWP(question, formula):
    q_tokens = tokenise_question(question)
    a_tokens = tokenise_formula(formula)

    numbers = []
    
    for i in range(len(q_tokens)):
        num = string_to_float(q_tokens[i])
        if num is not None:
            if num not in numbers:
                numbers.append(num)
                q_tokens[i] = "#" + str(len(numbers) - 1)
            else:
                print(q_tokens)
    
    for i in range(len(a_tokens)):
        num = string_to_float(a_tokens[i])
        if num is not None and num in numbers:
            a_tokens[i] = "#" + str(numbers.index(num))
    
    return q_tokens, a_tokens, numbers

def indexesFromTokens(lang, tokens):
    indexes = [lang.token2index[token] for token in tokens]
    indexes.append(EOS_token)
    return indexes

def tensorFromTokens(lang, tokens):
    indexes = indexesFromTokens(lang, tokens)
    tensor = torch.tensor(indexes, dtype=torch.long, device=device)
    return tensor

def tensorsFromPair(lang, pair):
    input_tensor = tensorFromTokens(lang, pair[0])
    target_tensor = tensorFromTokens(lang, pair[1])
    return (input_tensor, target_tensor)

def pad_indexes(indexes, max_length):
    indexes += [EOS_token for i in range(max_length - len(indexes))]
    return indexes

def indexesFromPairs(questions, formulas):
    all_tokens = [tokensFromMWP(question, formula) for question, formula in zip(questions, formulas)]
    q_indexes = [indexesFromTokens(q_lang, q) for q,_,_ in all_tokens]
    a_indexes = [indexesFromTokens(a_lang, a) for _,a,_ in all_tokens]
    numbers = [n for _,_,n in all_tokens]

    q_lengths = [len(q) for q in q_indexes]
    a_lengths = [len(a) for a in a_indexes]

    q_padded = [pad_indexes(q, max(q_lengths)) for q in q_indexes]
    a_padded = [pad_indexes(a, max(a_lengths)) for a in a_indexes]

    q_tensor = torch.LongTensor(q_padded, device=device).transpose(0, 1)
    a_tensor = torch.LongTensor(a_padded, device=device).transpose(0, 1)

    return q_tensor, a_tensor, q_lengths, a_lengths, numbers

def sort_by_length(sequences, lengths):
    restore_indexes = list(range(sequences.size(1)))

    sorted_indexes = torch.LongTensor(sorted(restore_indexes, key=lambda x: lengths[x], reverse=True), device=device)
    sorted_sequences = sequences.index_select(1, sorted_indexes)
    sorted_lengths = [lengths[i] for i in sorted_indexes]

    restore_indexes = torch.LongTensor(sorted(restore_indexes, key=lambda x: sorted_indexes[x], reverse=True), device=device)

    return sorted_sequences, sorted_lengths, restore_indexes
    