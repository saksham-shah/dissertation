import torch

from .attention import Attention, AttnDecoder
from .classifier import Classifier, AttnClassifier
from .decoder import Decoder
from .embedding import Embedding
from .encoder import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")