from models.embedding import Embedding
from models.encoder import Encoder
from models.decoder import Decoder
from models.attention import AttnDecoder
from data import *
from train import *
from config import *
from evaluate import *

# output = ""

# for dataset in ["mawps", "asdiv"]:
#     for rpn in [True, False]:
#         for num_emb in [True, False]:
#             config["dataset"] = dataset
#             config["rpn"] = rpn
#             config["num_emb"] = num_emb

# def run_experiment(config):
#     mwps, q_lang, a_lang = load_data(config)

#     embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
#     encoder = Encoder(config).to(device)
#     if config["attention"]:
#         decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
#     else:
#         decoder = Decoder(config, a_lang.n_tokens).to(device)

#     train_loader, test_loader = train_test(mwps)

#     max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 50, print_every=5)

#     # overall_acc = accuracy(config, test_loader, embedding, encoder, decoder, q_lang, a_lang)

#     print(acc)

#     return max_acc, acc, iters

def run_experiment(config, mwps, nfold=10):
    overall_acc = 0

    for fold in range(nfold):
        train_set, test_set = train_test(mwps, fold=fold, nfold=nfold)

        q_lang, a_lang = generate_vocab(config, train_set)

        train_loader = batch_data(train_set, config['rpn'], config['batch_size'])
        test_loader = batch_data(test_set, config['rpn'], 1)

        embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
        encoder = Encoder(config).to(device)
        if config["attention"]:
            decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
        else:
            decoder = Decoder(config, a_lang.n_tokens).to(device)

        max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 50, print_every=0)
        print(f"Fold: {fold}, Accuracy: {max_acc}")

        overall_acc += max_acc

    return overall_acc / nfold

options = {
    "batch_size": [1, 8],
    "dataset": ["asdiv", "mawps"],
}

# args = ["rpn", "num_embs", "batch_size"]

def get_config(base_config, args, exp_num):
    config = base_config.copy()
    for key in args:
        active = exp_num % 2
        exp_num = exp_num >> 1

        if key in options:
            config[key] = options[key][active]
        else:
            config[key] = active == 1
    return config

def run_experiments(base_config, args, mwps, nfold=10):
    output = ""
    args = args[::-1]
    for i in range(2 ** len(args)):
        config = get_config(base_config, args, i)
        acc = run_experiment(config, mwps, nfold=nfold)
        # max_acc, acc, iters = run_experiment(config, mwps, nfold=nfold)

        exp_str = format(i, f'0{len(args)}b')
        exp_output = f"{exp_str} - avg acc: {acc}\n"
        print(exp_output)

        output += exp_output
    print(output)

def save_model(embedding, encoder, decoder, q_lang, a_lang, path="model/"):
    torch.save(embedding, path + 'embedding.pt')
    torch.save(encoder, path + 'encoder.pt')
    torch.save(decoder, path + 'decoder.pt')

    with open(path + "token2index.json", "w") as file:
        json.dump(q_lang.token2index, file)

    with open(path + "index2token.json", "w") as file:
        json.dump(a_lang.index2token, file)

# run_experiment(config)

all_mwps, ids = load_data()

with open('data/test.txt') as file:
    test_ids = [line.rstrip() for line in file]

mwps = []
for mwp in all_mwps:
    if mwp not in test_ids:
        if config["dataset"] == "both" or config["dataset"] == mwp[:5]:
            mwps.append(all_mwps[mwp])

print(len(mwps))

random.shuffle(mwps)

run_experiments(config, ['rpn', 'attention', 'num_emb'], mwps, nfold=10)

# config['rpn'] = True

# train_set, test_set = train_test(mwps, 0, 2)

# q_lang, a_lang = generate_vocab(config, train_set)

# train_loader = batch_data(train_set, config['rpn'], config['batch_size'])
# test_loader = batch_data(test_set, config['rpn'], 1)

# embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
# encoder = Encoder(config).to(device)
# if config["attention"]:
#     decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
# else:
#     decoder = Decoder(config, a_lang.n_tokens).to(device)

# max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 1, print_every=10)
# print(f"Accuracy: {max_acc}")