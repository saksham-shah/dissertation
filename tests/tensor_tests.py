from utils.prepare_tensors import *

# Prepare tensors: indexes_from_tokens

def test_indexes_from_tokens(token2index, seq, indexes):
    tokens = seq.split(" ")
    result = indexes_from_tokens(token2index, tokens)
    assert result == indexes

token2index = {
    "SOS": 0,
    "EOS": 1,
    "Hello": 2,
    "world": 3,
    "!": 4,
    "UNK": 5,
}

indexes_from_tokens_tests = [
    ("Hello world !", [2, 3, 4, 1]),
    ("Hello , world !", [2, 5, 3, 4, 1]),
]

for seq, indexes in indexes_from_tokens_tests:
    test_indexes_from_tokens(token2index, seq, indexes)

# Prepare tensors: pad_indexes

def test_pad_indexes(array, max_length):
    assert len(array) <= max_length
    result = pad_indexes(array, max_length)
    assert len(result) == max_length
    assert array == result[:max_length]

pad_indexes_tests = [
    ([2, 3, 4, 5], 6),
    ([2, 3], 4),
    ([5, 6, 7, 8, 4], 5)
]

for test in pad_indexes_tests:
    test_pad_indexes(*test)

# Prepare tensors: sort_by_length

def test_sort_by_length():
    seqs = torch.tensor([
        [2, 3, 4, 5, 1, 1],
        [2, 3, 1, 1, 1, 1],
        [5, 6, 7, 8, 4, 1],
    ]).transpose(0, 1)

    seq_lengths = [4, 2, 5]

    target_sorted_seqs = torch.tensor([
        [5, 6, 7, 8, 4, 1],
        [2, 3, 4, 5, 1, 1],
        [2, 3, 1, 1, 1, 1],
    ]).transpose(0, 1)
    target_sorted_len = [5, 4, 2]
    target_restore_indexes = torch.tensor([1, 2, 0])

    sorted_seqs, sorted_len, restore_indexes = sort_by_length(seqs, seq_lengths)
    assert torch.all(sorted_seqs == target_sorted_seqs)
    assert sorted_len == target_sorted_len
    assert torch.all(restore_indexes == target_restore_indexes)

test_sort_by_length()

print("All tests passed.")