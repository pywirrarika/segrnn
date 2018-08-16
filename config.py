# Constants from C++ code
EMBEDDING_DIM = 64 + 1
LAYERS_1 = 2
LAYERS_2 = 1
INPUT_DIM = 64 + 1
XCRIBE_DIM = 24
SEG_DIM = 24
H1DIM = 32
H2DIM = 32
TAG_DIM = 32
DURATION_DIM = 4
DROPOUT = 0.0

# lstm builder: LAYERS, XCRIBE_DIM, SEG_DIM, m?
# (layers, input_dim, hidden_dim, model)

DATA_MAX_SEG_LEN = 15

MAX_SENTENCE_LEN = 32
MINIBATCH_SIZE = 64
BATCH_SIZE = 256

use_max_sentence_len_training = True
use_bucket_training = False

#LABELS = ['DET', 'AUX', 'ADJ', 'ADP', 'VERB', 'NOUN', 'SYM', 'PROPN', 'PART', 'X', 'CCONJ', 'PRON', 'ADV', 'PUNCT', 'NUM', 'BLANK']
LABELS = ['TR', 'DE', 'LANG3', 'MIXED', 'OTHER', 'NE', 'NE.TR', 'NE.MIXED', 'NE.DE', 'AMBIGuous',]


