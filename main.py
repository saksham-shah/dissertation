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

def save_model(embedding, encoder, decoder, q_lang, a_lang, path="model/"):
    torch.save(embedding, path + 'embedding.pt')
    torch.save(encoder, path + 'encoder.pt')
    torch.save(decoder, path + 'decoder.pt')

    with open(path + "token2index.json", "w") as file:
        json.dump(q_lang.token2index, file)

    with open(path + "index2token.json", "w") as file:
        json.dump(a_lang.index2token, file)

mwps = prepare_training_data(config['dataset'])

print(len(mwps))

folds_avg, folds = run_experiments(config, ['rpn', 'attention', 'num_emb'], mwps, nfold=9)

t_stat, is_better = paired_t_test(folds)

np.savetxt('folds.csv', folds, delimiter=',')
np.savetxt('t_stat.csv', t_stat, delimiter=',')
np.savetxt('is_better.csv', is_better, delimiter=',')