from experiment import *
from copy import deepcopy

config = {
    "num_layers": 1,
    "batch_size": 1,
    "teacher_forcing_ratio": 0.9,
    "learning_rate": 0.0005,
    "hidden_size": 256,
    "bidirectional": True,
    "dropout": 0.1,
    "early_stopping": -1,
    "rpn": True,
    "num_emb": True,
    "embedding_size": 300,
    "dataset": ["asdiv", "mawps"],
    "attention": True,
}

# Load all data
def prepare_data():
    print("Loading data")
    all_mwps, ids = load_data()

    # Split into train and test
    print("Splitting data")
    with open('data/test.txt') as file:
        test_ids = [line.rstrip() for line in file]

    train_mwps = []
    test_mwps = []
    for mwp in all_mwps:
        if mwp[:5] in ["asdiv", "mawps"]:
            if mwp in test_ids:
                test_mwps.append(all_mwps[mwp])
            else:
                train_mwps.append(all_mwps[mwp])

    print(f"# train: {len(train_mwps)}, # test: {len(test_mwps)}")

    # train_mwps = train_mwps[:10]
    # test_mwps = test_mwps[:10]

    # random.shuffle(train_mwps)
    # random.shuffle(test_mwps)
    return train_mwps, test_mwps

# Train seq2seq models
def train_seq2seq(config, train_set, test_set):
    train_set = deepcopy(train_set)
    q_lang, a_lang = generate_vocab(config, train_set)

    train_loader = batch_data(train_set, config['rpn'], 1) # config['batch_size']
    test_loader = batch_data(test_set, config['rpn'], 1)

    embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
    encoder = Encoder(config).to(device)
    if config["attention"]:
        decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
    else:
        decoder = Decoder(config, a_lang.n_tokens).to(device)

    max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 50, print_every=0)
    print(f"Accuracy: {max_acc}")
    return embedding, encoder, decoder, q_lang, a_lang

# Train improved model
train_mwps, test_mwps = prepare_data()

asdiv_test = []
mawps_test = []
for mwp in test_mwps:
    if mwp.id[:5] == 'asdiv':
        asdiv_test.append(mwp)
    elif mwp.id[:5] == 'mawps':
        mawps_test.append(mwp)
asdiv_loader = batch_data(asdiv_test, config['rpn'], 1)
mawps_loader = batch_data(mawps_test, config['rpn'], 1)
combined_loader = batch_data(test_mwps, config['rpn'], 1)

print(f"TEST: # total = {len(test_mwps)}, # asdiv = {len(asdiv_test)}, # mawps = {len(mawps_test)}")

asdiv_mwps = []
mawps_mwps = []
for mwp in train_mwps:
    if mwp.id[:5] == 'asdiv':
        asdiv_mwps.append(mwp)
    elif mwp.id[:5] == 'mawps':
        mawps_mwps.append(mwp)
print(f"TRAIN: # total = {len(train_mwps)}, # asdiv = {len(asdiv_mwps)}, # mawps = {len(mawps_mwps)}")

print("ASDiv")

embedding, encoder, decoder, q_lang, a_lang = train_seq2seq(config, asdiv_mwps, test_mwps)

asdiv_acc = accuracy(config, asdiv_loader, embedding, encoder, decoder, q_lang, a_lang)
mawps_acc = accuracy(config, mawps_loader, embedding, encoder, decoder, q_lang, a_lang)
combined_acc = accuracy(config, combined_loader, embedding, encoder, decoder, q_lang, a_lang)

print(f"ASDiv: {asdiv_acc}, MAWPS: {mawps_acc}, Combined: {combined_acc}")

print("Saving ASDiv")
torch.save({
    'embedding': embedding,
    'encoder': encoder,
    'decoder': decoder,
    'token2index': q_lang.token2index,
    'index2token': a_lang.index2token,
}, 'final/asdiv.pt')

# Train MAWPS model
print("MAWPS")
# train_mwps, test_mwps = prepare_data()

# mawps_mwps = []
# for mwp in train_mwps:
#     if mwp.id[:5] == 'mawps':
#         mawps_mwps.append(mwp)
# print(f"# mawps train = {len(mawps_mwps)}")

embedding, encoder, decoder, q_lang, a_lang = train_seq2seq(config, mawps_mwps, test_mwps)

asdiv_acc = accuracy(config, asdiv_loader, embedding, encoder, decoder, q_lang, a_lang)
mawps_acc = accuracy(config, mawps_loader, embedding, encoder, decoder, q_lang, a_lang)
combined_acc = accuracy(config, combined_loader, embedding, encoder, decoder, q_lang, a_lang)

print(f"ASDiv: {asdiv_acc}, MAWPS: {mawps_acc}, Combined: {combined_acc}")

print("Saving MAWPS")
torch.save({
    'embedding': embedding,
    'encoder': encoder,
    'decoder': decoder,
    'token2index': q_lang.token2index,
    'index2token': a_lang.index2token,
}, 'final/mawps.pt')

# Test combined
print("Combined")
# train_mwps, test_mwps = prepare_data()

# print(f"# combined train = {len(train_mwps)}")
embedding, encoder, decoder, q_lang, a_lang = train_seq2seq(config, train_mwps, test_mwps)

asdiv_acc = accuracy(config, asdiv_loader, embedding, encoder, decoder, q_lang, a_lang)
mawps_acc = accuracy(config, mawps_loader, embedding, encoder, decoder, q_lang, a_lang)
combined_acc = accuracy(config, combined_loader, embedding, encoder, decoder, q_lang, a_lang)

print(f"ASDiv: {asdiv_acc}, MAWPS: {mawps_acc}, Combined: {combined_acc}")

print("Saving combined")
torch.save({
    'embedding': embedding,
    'encoder': encoder,
    'decoder': decoder,
    'token2index': q_lang.token2index,
    'index2token': a_lang.index2token,
}, 'final/combined.pt')