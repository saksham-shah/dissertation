import math
import random

import torch
import transformers
print(transformers.__version__)

from data import *

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model_checkpoint = "facebook/bart-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "rpn": True,
    "dataset": "mawps",
    "epochs": 5,
    "weight_decay": 0.01,
}

def mwp_to_dict(mwp):
    return {
        "id": mwp.id,
        "question": mwp.question,
        "equation": mwp.equation,
        "answer": mwp.answer,
    }

def train_test_split(data, test_size=0.1):
    random.seed(1)
    random.shuffle(data)

    boundary = math.floor(len(data) * (1 - test_size))
    train = data[:boundary]
    test = data[boundary:]
    return { "train": train, "test": test }

class MWPDataset(torch.utils.data.Dataset):
  def __init__(self, inputs, targets):
    self.inputs = inputs
    self.targets = targets

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx], device=device) for key, val in self.inputs.items()}
    item['labels'] = torch.tensor(self.targets['input_ids'][idx], device=device)
    return item
  
  def __len__(self):
    return len(self.inputs['input_ids'])
  
def get_data(config):
    mwps, _, _ = load_data(config)
    data = list(map(mwp_to_dict, mwps))

    inputs = train_test_split([mwp["question"] for mwp in data])
    targets = train_test_split([mwp["equation"] for mwp in data])

    return inputs, targets

def tokenise_data(tokeniser, inputs, targets):
    max_input_length = 1024
    max_target_length = 64

    tokenised_inputs = {
        "train": tokeniser(inputs["train"], max_length=max_input_length, truncation=True),
        "test": tokeniser(inputs["test"], max_length=max_input_length, truncation=True),
    }

    tokenised_targets = {
        "train": tokeniser(targets["train"], max_length=max_target_length, truncation=True),
        "test": tokeniser(targets["test"], max_length=max_target_length, truncation=True),
    }

    train_dataset = MWPDataset(tokenised_inputs["train"], tokenised_targets["train"])
    test_dataset = MWPDataset(tokenised_inputs["test"], tokenised_targets["test"])

    return train_dataset, test_dataset

def train_model(config, model, tokeniser, train_dataset, test_dataset):
    batch_size = config["batch_size"]
    args = Seq2SeqTrainingArguments(
        f"{model_checkpoint}-finetunes-mawps",
        evaluation_strategy = "epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=config["weight_decay"],
        save_total_limit=3,
        num_train_epochs=config["epochs"],
        predict_with_generate=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokeniser, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokeniser,
    )

    print("Training now...")

    trainer.train()

def evaluate_accuracy(model, tokeniser, inputs, targets):
    correct = 0
    for i in range(len(inputs)):
        input = inputs[i]
        target = targets[i]

        input_tokens = tokeniser([input], max_length=1024, return_tensors='pt')
        input_tokens['input_ids'].to(device)

        pred_tokens = model.generate(input_tokens['input_ids'], num_beams=4, max_length=32, early_stopping=True)
        pred = [tokeniser.decode(token, skip_special_tokens=True, clean_up_tokenization_spaces=False) for token in pred_tokens]

        if pred == target:
            print(pred)
            correct += 1

    return correct / len(inputs)

tokeniser = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

inputs, targets = get_data(config)
train_dataset, test_dataset = tokenise_data(tokeniser, inputs, targets)

train_model(config, model, tokeniser, train_dataset, test_dataset)