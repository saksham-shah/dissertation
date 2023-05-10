# from models.embedding import Embedding
# from models.encoder import Encoder
# from models.decoder import Decoder
# from models.attention import AttnDecoder
# from data import *
# from train import *
# from evaluate import *
from evaluation.experiment import *

config = {
    "num_layers": 1,
    "batch_size": 1,
    "teacher_forcing_ratio": 0.9,
    "learning_rate": 0.0005,
    "hidden_size": 256,
    "bidirectional": True,
    "dropout": 0.1,
    "early_stopping": 5,
    "rpn": False,
    "num_emb": True,
    "embedding_size": 300,
    "dataset": ["asdiv", "mawps"],
    "attention": True,
}

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

# def run_experiment(config, mwps, nfold=10):
#     overall_acc = 0

#     for fold in range(nfold):
#         train_set, test_set = train_test(mwps, fold=fold, nfold=nfold)

#         q_lang, a_lang = generate_vocab(config, train_set)

#         train_loader = batch_data(train_set, config['rpn'], config['batch_size'])
#         test_loader = batch_data(test_set, config['rpn'], 1)

#         embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
#         encoder = Encoder(config).to(device)
#         if config["attention"]:
#             decoder = AttnDecoder(config, a_lang.n_tokens).to(device)
#         else:
#             decoder = Decoder(config, a_lang.n_tokens).to(device)

#         max_acc, acc, iters = trainIters(config, train_loader, test_loader, embedding, encoder, decoder, q_lang, a_lang, 50, print_every=0)
#         print(f"Fold: {fold}, Accuracy: {max_acc}")

#         overall_acc += max_acc

#     return overall_acc / nfold

# options = {
#     "batch_size": [1, 8],
#     "dataset": ["asdiv", "mawps"],
# }

# # args = ["rpn", "num_embs", "batch_size"]

# def get_config(base_config, args, exp_num):
#     config = base_config.copy()
#     for key in args:
#         active = exp_num % 2
#         exp_num = exp_num >> 1

#         if key in options:
#             config[key] = options[key][active]
#         else:
#             config[key] = active == 1
#     return config

# def run_experiments(base_config, args, mwps, nfold=10):
#     output = ""
#     args = args[::-1]
#     for i in range(2 ** len(args)):
#         config = get_config(base_config, args, i)
#         acc = run_experiment(config, mwps, nfold=nfold)
#         # max_acc, acc, iters = run_experiment(config, mwps, nfold=nfold)

#         exp_str = format(i, f'0{len(args)}b')
#         exp_output = f"{exp_str} - avg acc: {acc}\n"
#         print(exp_output)

#         output += exp_output
#     print(output)

def save_model(embedding, encoder, decoder, q_lang, a_lang, path="model/"):
    torch.save(embedding, path + 'embedding.pt')
    torch.save(encoder, path + 'encoder.pt')
    torch.save(decoder, path + 'decoder.pt')

    with open(path + "token2index.json", "w") as file:
        json.dump(q_lang.token2index, file)

    with open(path + "index2token.json", "w") as file:
        json.dump(a_lang.index2token, file)

# # run_experiment(config)

# all_mwps, ids = load_data()

# with open('data/test.txt') as file:
#     test_ids = [line.rstrip() for line in file]

# mwps = []
# for mwp in all_mwps:
#     if mwp not in test_ids:
#         if mwp[:5] in config["dataset"]:
#             mwps.append(all_mwps[mwp])

# print(len(mwps))

# random.shuffle(mwps)

mwps = prepare_training_data(config['dataset'])

print(len(mwps))

folds_avg, folds = run_experiments(config, {
    "hidden_size": [64, 128, 256, 512],
    "batch_size": [1, 2],
}, mwps, nfold=9, binary=False)
# # folds_avg, folds = run_experiments(config, ['rpn'], mwps, nfold=9)
# folds = np.loadtxt('results/hyperparameter/folds.csv', delimiter=',')
# t_stat, is_better = paired_t_test(folds)

# np.savetxt('folds.csv', folds, delimiter=',')
# np.savetxt('t_stat.csv', t_stat, delimiter=',')
# np.savetxt('is_better.csv', is_better, delimiter=',')

is_better = np.loadtxt('results/hyperparameter/is_better.csv', delimiter=',')
folds = np.loadtxt('results/hyperparameter/folds.csv', delimiter=',')
t_stat = np.loadtxt('results/hyperparameter/t_stat.csv', delimiter=',')
print(is_better)
# print(np.loadtxt('results/asdiv/is_better.csv', delimiter=','))
# print(t_stat > 2.306)


def accuracy(config, test_set):
    correct = 0
    incorrect = 0
    for mwp in test_set:
        # q_tokens, a_tokens, numbers = tokens_from_MWP(mwp.question, mwp.equation)
        a_tokens = mwp['formula'][0].split(" ")
        numbers = list(map(float, mwp['numbers'][0].split(",")))

        if check(config, a_tokens, a_tokens, mwp['answer'][0], numbers):
            correct += 1
        else:
            incorrect += 1
            print(mwp)
    print(f"Incorrect: {incorrect}")
    print("Accuracy:", correct / len(test_set))
    return correct / len(test_set)



print(len(mwps))

train_set, test_set = train_test(mwps, 0, 2)
train_loader = batch_data(mwps, config['rpn'], 1)

print(len(train_loader))

accuracy(config, train_loader)

# print(folds)

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