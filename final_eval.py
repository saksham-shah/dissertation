from experiment import *
from bart import *
from train_classifier import *

config = {
    "num_layers": 1,
    "batch_size": 1,
    "teacher_forcing_ratio": 0.9,
    "learning_rate": 0.0005,
    "hidden_size": 256,
    "bidirectional": True,
    "dropout": 0.1,
    "early_stopping": -1,
    "rpn": False,
    "num_emb": False,
    "embedding_size": 300,
    "dataset": ["asdiv", "mawps"],
    "attention": False,
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

# Train vanilla model
print("Vanilla seq2seq")
train_mwps, test_mwps = prepare_data()
embedding, encoder, decoder, q_lang, a_lang = train_seq2seq(config, train_mwps, test_mwps)
print("Saving baseline")
torch.save({
    'embedding': embedding,
    'encoder': encoder,
    'decoder': decoder,
    'token2index': q_lang.token2index,
    'index2token': a_lang.index2token,
}, 'final/baseline.pt')

# Train improved model
print("Improved seq2seq")
train_mwps, test_mwps = prepare_data()
config['rpn'] = True
config['attention'] = True
config['num_embs'] = True
embedding, encoder, decoder, q_lang, a_lang = train_seq2seq(config, train_mwps, test_mwps)
print("Saving improved")
torch.save({
    'embedding': embedding,
    'encoder': encoder,
    'decoder': decoder,
    'token2index': q_lang.token2index,
    'index2token': a_lang.index2token,
}, 'final/improved.pt')

# Train BART
print("BART")
train_mwps, test_mwps = prepare_data()
train_data = list(map(mwp_to_dict, train_mwps))
test_data = list(map(mwp_to_dict, test_mwps))

inputs = {
    'train': [mwp["question"] for mwp in train_data],
    'test': [mwp["question"] for mwp in test_data],
}

targets = {
    'train': [mwp["equation"] for mwp in train_data],
    'test': [mwp["equation"] for mwp in test_data],
}

tokeniser = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)#.to(device)

train_dataset, test_dataset = tokenise_data(tokeniser, inputs, targets)

trainer = train_model(config, model, tokeniser, train_dataset, test_dataset, test_data)

print("Saving BART")
trainer.save_model('final/bart')

# print(evaluate_accuracy(model, tokeniser, inputs['test'], targets['test'], mwps['test']))

# Train classifier
print("Classifier")
train_mwps, test_mwps = prepare_data()
q_lang, a_lang = generate_vocab(config, train_mwps)

# print(q_lang.token2index)

train_loader = batch_data(train_mwps, True, 1) # config['batch_size']
test_loader = batch_data(test_mwps, True, 1)

# embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
classifier = Classifier(config, q_lang.n_tokens, a_lang.n_tokens, q_lang, a_lang).to(device)

optimiser = torch.optim.Adam(classifier.parameters(), lr=config['learning_rate'])
criterion = torch.nn.BCELoss()

force_correct = 0.5

NUM_EPOCHS = 50

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train(config, classifier, train_loader, optimiser, criterion, q_lang, a_lang)
    val_loss = evaluate(config, classifier, test_loader, q_lang, a_lang)
    # print(f"Epoch {epoch}: loss={train_loss}")
    end_time = timer()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val acc: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

print("Saving classifier")
torch.save(classifier, 'final/classifier.pt')