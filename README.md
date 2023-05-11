This is the code repository for my Part II project: Building a Model to Solve and Explain Math Word Problems.

The goal of the project was to build an automatic Math Word Problem (MWP) solver, which would take a question as input and output an equation which could be evaluated to compute the correct answer. The project aimed to first build a baseline encoder-decoder model and then modify the model to improve its performance.

As an extension, the project also aimed to fine-tune a pre-trained transformer to the task of solving MWPs, as well as implement an alternative classifier-based approach, and then compare the performance of all the models implemented.

`seq2seq` contains code to train and evaluate the encoder-decoder model. `models` contains all of the PyTorch model definitions. `data` contains the ASDiv and MAWPS data used, as well as code to load this data. `user_interaction` contains code to allow the user to use a trained model to solve MWPs. `other_models` contains code related to the extensions, to fine-tune BART and implement a binary classifier. `evaluation` contains code to run the experiments discussed in the Evaluation chapter. `utils` contains various utility functions to preprocess the data and `tests` contains unit tests.

Dependencies:
- PyTorch and torchtext
- HuggingFace transformers
- word2number