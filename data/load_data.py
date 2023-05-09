import xml.etree.ElementTree as ET
import json
import re
from torchtext.vocab import GloVe
import numpy as np
from utils.process_input import *
from dataclasses import dataclass

global_vectors = GloVe(name='6B', dim=300)

@dataclass
class MWP:
    id: str
    question: str
    equation: str
    answer: float
    numbers: str

# Create MWP object from question and equation strings
def create_MWP(id, question, equation, answer):
    q_tokens, a_tokens, numbers = tokensFromMWP(question, equation)

    # Filter out invalid MWPs
    if q_tokens is None:
        return None
    
    return MWP(id, " ".join(q_tokens), " ".join(a_tokens), answer, ",".join(map(str, numbers)))

# Create MWP object from XML (ASDiv)
def MWP_from_asdiv(xml_problem):
    id = xml_problem.attrib['ID']
    body = xml_problem.find('Body').text
    question = xml_problem.find('Question').text
    solution_type = xml_problem.find('Solution-Type').text
    answer = xml_problem.find('Answer').text
    formula = xml_problem.find('Formula').text

    # Filter by category
    if solution_type not in ['Addition', 'Subtraction', 'Multiplication', 'Common-Division', 'Sum', 'Difference', 'TVQ-Initial', 'TVQ-Change', 'TVQ-Final']:
        return None
        
    # Filter out MWPs requiring domain knowledge of units
    solution_attrib = xml_problem.find('Solution-Type').attrib
    if 'UnitTrans' in solution_attrib:
        return None
    
    id = "asdiv" + str(id.split("-")[1])

    equation = formula.split('=')[0]
    answer = float(re.sub(r" \(.+\)", r"", answer))

    question = body + question
    question = re.sub(r"([?$])", r" \1 ", question) # add spaces before/after ? and $ characters
    question = re.sub(r"([.,])([^0-9])", r" \1 \2", question) # add spaces before/after . and , characters IF NOT PRECEEDING NUMBER

    return create_MWP(id, question, equation, answer)

# Create MWP object from JSON (MAWPS)
def MWP_from_mawps(mawps):
    id = "mawps" + str(mawps["id"])
    question = mawps["original_text"] # already tokenised in dataset
    answer = float(mawps["ans"])

    # Filter complex equations
    if mawps["equation"].lower()[:2] != 'x=':
        return None

    equation = mawps["equation"].split('=')[1]

    return create_MWP(id, question, equation, answer)

SOS_token = 0 # start-of-sequence
EOS_token = 1 # end-of-sequence

# Maintains input/output vocabularies
class Lang:
    def __init__(self):
        self.token2index = {}
        self.token2count = {}
        self.index2token = {0: 'SOS', 1: 'EOS'}
        self.n_tokens = 2
        self.weights = None

    def addTokens(self, tokens):
        for token in tokens:
            self.addToken(token)
    
    def addToken(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1
    
    def set_weights(self, weights):
        self.weights = weights

# Loads data from XML and JSON files
# Returns list of MWP objects and their IDs
def load_data():
    mwps = {}
    ids = []

    # Load ASDiv
    with open('./data/asdiv/asdiv_a.txt') as file:
        asdiv_a_ids = [line.rstrip() for line in file]
    
    tree = ET.parse('./data/asdiv/asdiv.xml')
    root = tree.getroot()

    for child in root:
        if child.attrib['ID'] in asdiv_a_ids:
            mwp = MWP_from_asdiv(child)
            if mwp is not None:
                mwps[mwp.id] = mwp
                ids.append(mwp.id)
    
    # Load MAWPS
    with open('./data/mawps/mawps.json') as file:
        mawps = json.loads(file.read())

    for mawps_q in mawps:
        mwp = MWP_from_mawps(mawps_q)
        if mwp is not None:
            mwps[mwp.id] = mwp
            ids.append(mwp.id)

    return mwps, ids

# Generates input/output vocabularies
# Returns Lang objects for input and output
def generate_vocab(config, mwps):
    q_lang = Lang() # input (q - question)
    a_lang = Lang() # output (a - answer)

    # Count word frequencies
    tokens = {}
    for mwp in mwps:
        for token in mwp.question.split(" "):
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1
    
    for mwp in mwps:
        unk_updated = []
        for token in mwp.question.split(" "):
            # Replace low freq words with UNK (unknown token)
            if tokens[token] > 1 or token[0] == '#':
                unk_updated.append(token)
            else:
                unk_updated.append("UNK")
        mwp.question = " ".join(unk_updated)

    # Generate Lang objects
    for mwp in mwps:
        q_lang.addTokens(mwp.question.split(" "))
        a_lang.addTokens(mwp.equation.split(" "))

    # Construct word embedding matrix for torch.nn.Embedding
    if "embedding_size" in config:
        embedding_size = config["embedding_size"]

        q_weights_matrix = np.zeros((q_lang.n_tokens, embedding_size))
        a_weights_matrix = np.zeros((a_lang.n_tokens, embedding_size))

        # Input embeddings
        for i, token in enumerate(q_lang.token2index):
            embedding = global_vectors.get_vecs_by_tokens([token], lower_case_backup=True)[0]
            if embedding.norm() != 0:
                q_weights_matrix[i] = embedding
            else:
                # Initialise random vector for words without an embedding
                q_weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_size, ))

        # Repeat for output embeddings
        for i, token in enumerate(a_lang.token2index):
            embedding = global_vectors.get_vecs_by_tokens([token], lower_case_backup=True)[0]
            if embedding.norm() != 0:
                a_weights_matrix[i] = embedding
            else:
                a_weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_size, ))
        
        q_lang.set_weights(q_weights_matrix)
        a_lang.set_weights(a_weights_matrix)
    
    return q_lang, a_lang