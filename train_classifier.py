import random
from data import *
from config import *
from utils.load_batches import *
import torch
from models.classifier import Classifier
from utils.prepare_tensors import *
from timeit import default_timer as timer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_mwps, ids = load_data()

with open('data/test.txt') as file:
    test_ids = [line.rstrip() for line in file]

mwps = []
for mwp in all_mwps:
    if mwp not in test_ids:
        if mwp[:5] in config["dataset"]:
            mwps.append(all_mwps[mwp])

print(len(mwps))

random.shuffle(mwps)

valid_mwps = []

for mwp in mwps:
    if len(mwp.numbers.split(",")) <= 3:
        if len(mwp.equation.split(" ")) <= 3:
            valid_mwps.append(mwp)

print(len(valid_mwps))
# mwps = valid_mwps

def get_all_equations():
    max_numbers = 3
    operations = ["+", "-", "*", "/"]

    all_pairs = []

    for i in range(max_numbers):
        for j in range(max_numbers):
            if i == j:
                continue
            # all_pairs.append(["#" + str(i), "#" + str(j)])
            all_pairs.append(f"#{str(i)} #{str(j)} ")

    all_equations = []

    for pair in all_pairs:
        for operation in operations:
            all_equations.append(pair + operation)

    return all_equations

def is_equivalent(e1, e2):
    if e1 == e2:
        return True
    
    tokens1 = e1.split(" ")
    tokens2 = e2.split(" ")

    if len(tokens1) != len(tokens2):
        return False

    # Check if operators are the same
    # Assume RPN (e.g. "#0 #1 +") so use index 2
    if tokens1[2] != tokens2[2]:
        return False
    
    # Commutativity of addition/multiplication
    if tokens1[2] in ["+", "*"]:
        return (tokens1[0] == tokens2[1]) and (tokens2[0] == tokens1[1])

all_equations = get_all_equations()

train_set, test_set = train_test(valid_mwps)

config['rpn'] = True
config['dataset'] = ['asdiv']

q_lang, a_lang = generate_vocab(config, train_set)

train_loader = batch_data(train_set, config['rpn'], config['batch_size'])
test_loader = batch_data(test_set, config['rpn'], 1)

# embedding = Embedding(config, q_lang.n_tokens, q_lang).to(device)
classifier = Classifier(config, q_lang.n_tokens, a_lang.n_tokens, q_lang, a_lang).to(device)

optimiser = torch.optim.Adam(classifier.parameters(), lr=config['learning_rate'])
criterion = torch.nn.BCELoss()

force_correct = 0.5

def evaluate(config, model, test_loader, q_lang, a_lang):
    correct = 0

    for mwp in test_loader:
        q_tensor = tensorFromTokens(q_lang.token2index, mwp['question'][0].split(" ")).view(-1, 1)
        numbers = list(map(float, mwp['numbers'][0].split(",")))

        max_output = 0
        best_equation = None

        for equation in all_equations:

            a_tensor = tensorFromTokens(a_lang.token2index, equation.split(" ")).view(-1, 1)
            output = model(q_tensor, a_tensor, [q_tensor.size(0)], [a_tensor.size(0)], [numbers])

            if output > max_output:
                max_output = output
                best_equation = equation
        
        if is_equivalent(best_equation, mwp['formula'][0]):
            correct += 1
            
    return correct / len(list(test_loader))

def train(config, model, train_loader, optimiser, criterion, q_lang, a_lang):
    model.train()

    losses = 0

    count = 0

    for mwp in train_loader:
        q_indexes = [indexesFromTokens(q_lang.token2index, q.split(" ")) for q in mwp['question']]
        q_lengths = [len(q) for q in q_indexes]
        q_padded = [pad_indexes(q, max(q_lengths)) for q in q_indexes]
        q_tensor = torch.tensor(q_padded, device=device).transpose(0, 1)

        numbers = [list(map(float, nums.split(","))) for nums in mwp['numbers']]

        for equation in all_equations:

            a_strings = [a if random.random() < force_correct else equation for a in mwp['formula']]
            target = [1.0 if is_equivalent(a, equation) else 0.0 for a in a_strings]
            
            a_indexes = [indexesFromTokens(a_lang.token2index, a.split(" ")) for a in a_strings]
            a_lengths = [len(a) for a in a_indexes]
            a_padded = [pad_indexes(a, max(a_lengths)) for a in a_indexes]
            a_tensor = torch.tensor(a_padded, device=device).transpose(0, 1)

            output = model(q_tensor, a_tensor, q_lengths, a_lengths, numbers)

            optimiser.zero_grad()

            loss = criterion(output, torch.tensor(target, device=device))
            loss.backward()

            optimiser.step()
            losses += loss.item()

            if count % 1000 == 0:
                print(f"Progress={(count * 100 / len(list(train_loader)) / len(all_equations)):.3f}%, loss={loss.item()}")

            count += 1

    return losses / len(list(train_loader)) / len(all_equations)

NUM_EPOCHS = 50

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train(config, classifier, train_loader, optimiser, criterion, q_lang, a_lang)
    val_loss = evaluate(config, classifier, test_loader, q_lang, a_lang)
    # print(f"Epoch {epoch}: loss={train_loss}")
    end_time = timer()
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val acc: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

torch.save(classifier, 'classifier-final.pt')