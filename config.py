import torch

# Constants from C++ code
EMBEDDING_DIM = 300 + 1
LAYERS_1 = 2
LAYERS_2 = 1
INPUT_DIM = 300 + 1
XCRIBE_DIM = 24
SEG_DIM = 24
H1DIM = 32
H2DIM = 32
TAG_DIM = 32
DURATION_DIM = 4
DROPOUT = 0.1
EPOCHS = 2000
EVAL_EACH_EPOCH = 20
# lstm builder: LAYERS, XCRIBE_DIM, SEG_DIM, m?
# (layers, input_dim, hidden_dim, model)

DATA_MAX_SEG_LEN = 15

MAX_SENTENCE_LEN = 32
MINIBATCH_SIZE = 300
BATCH_SIZE = 256

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

use_max_sentence_len_training = True
use_bucket_training = False


prefix = "srnn_model"

LABELS = [
    'DET', 'AUX', 'ADJ', 'ADP', 'VERB', 'NOUN', 'SYM', 'PROPN', 'PART', 'X', 'CCONJ', 'PRON', 'ADV', 'PUNCT', 'NUM', 'BLANK', 'SCONJ', '_', #POS TAGS
    'AMBIG', 'TR', 'DE', 'LANG3', 'MIXED', 'OTHER', 'NE', 'NE.OTHER', 'NE.TR', 'NE.MIXED', 'NE.DE', 'NE.LANG3', 'NE.AMBIG', 'AMBIGuous', #LANGID TAGS
    'UNK'
    ]
