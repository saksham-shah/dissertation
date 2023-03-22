import math
import random

import transformers
print(transformers.__version__)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from data import *

model_checkpoint = "facebook/bart-base"

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

data = train_test_split(list(map(mwp_to_dict, mwps)))

print(data["train"][0])


tokeniser = AutoTokenizer.from_pretrained(model_checkpoint)

print(tokeniser("Hello, this one sentence!"))

print(Q_MAX_LENGTH, A_MAX_LENGTH)

max_input_length = 1024
max_target_length = 64

# def preprocess_function(data):
#     # print("preprocess: ")
#     # for mwp in data:
#     #     print(data)
#     #     print(mwp)
#     inputs = [mwp["question"] for mwp in data]
#     targets = [mwp["equation"] for mwp in data]
#     model_inputs = tokeniser(inputs, max_length=max_input_length, truncation=True)

#     # Setup the tokenizer for targets
#     with tokeniser.as_target_tokenizer():
#         labels = tokeniser(targets, max_length=max_target_length, truncation=True)

#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

def preprocess_function(data):
    # print("preprocess: ")
    # for mwp in data:
    #     print(data)
    #     print(mwp)
    inputs = [data["question"]]
    targets = [data["equation"]]
    model_inputs = tokeniser(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokeniser.as_target_tokenizer():
        labels = tokeniser(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# print(preprocess_function(data["train"][:2]))

# print("Data:")
# print(data["train"])

tokenised_data = {
    "train": list(map(preprocess_function, data["train"])),
    "test": list(map(preprocess_function, data["test"])),
}

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
args = Seq2SeqTrainingArguments(
    f"{model_checkpoint}-finetunes-mawps",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
)

data_collator = DataCollatorForSeq2Seq(tokeniser, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenised_data["train"],
    eval_dataset=tokenised_data["test"],
    data_collator=data_collator,
    tokenizer=tokeniser,
)

print("Training now...")



trainer.train()
