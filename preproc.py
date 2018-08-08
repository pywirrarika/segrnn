import numpy as np
from config import *

def parse_embedding(embed_filename):
    embed_file = open(embed_filename)
    embedding = dict()
    for line in embed_file:
        values = line.split()
        values.append(1.0)
        embedding[values[0]] = np.array(values[1:]).astype(np.float)
    return embedding

def parse_file(train_filename, embedding, use_max_len=True):
    train_file = open(train_filename)
    sentences = []
    labels = []
    label = []
    POS_labels = set()
    sentence = ""
    label_sum = 0
    for line in train_file:
        if line.startswith("# text = "):
            sentence = line[9:].strip().replace(" ", "")
            N = len(sentence)
            if use_max_len:
                max_len = MAX_SENTENCE_LEN
            else:
                max_len = N
            sentence_vec = np.zeros((max_len, EMBEDDING_DIM))
            for i in range(min(N, max_len)):
                c = sentence[i]
                if c in embedding:
                    sentence_vec[i, :] = embedding[c]
                elif c in "0123456789":
                    sentence_vec[i, :] = embedding["<NUM>"]
                else:
                    sentence_vec[i, :] = embedding["<unk>"]
            sentences.append(sentence_vec)
        elif not line.startswith("#"):
            parts = line.split()
            if len(parts) < 4:
                if len(sentence) != 0:
                    while label_sum < max_len:
                        label_len = 1
                        label_sum += label_len
                        label.append(('BLANK', label_len))
                    labels.append((label, sentence))
                label = []
                label_sum = 0
                sentence = ""
            else:
                if (label_sum + len(parts[1])) <= max_len:
                    label_sum += len(parts[1])
                    label.append((parts[3], len(parts[1])))
                else:
                    label_len = max_len - label_sum
                    if label_len > 0:
                        label.append((parts[3], label_len))
                        label_sum = max_len

    return sentences, labels


