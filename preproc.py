import numpy as np
import sys
from config import *

def parse_embedding(embed_filename):

    print("Parsing embedding file")
    embed_file = open(embed_filename)
    embedding = dict()
    for line in embed_file:
        values = line.split()
        values.append(1.0)
        embedding[values[0]] = np.array(values[1:]).astype(np.float)
    return embedding

# Fake embeddings. We use this function if not character embeddings 
# are available 
def parse_embedding_fake(embed_filename):
    
    print("Generating random embeddings")
    embedding = dict()
    embedding['<unk>'] = np.random.uniform(-1,0,EMBEDDING_DIM)
    embedding['<NUM>'] = np.random.uniform(-1,0,EMBEDDING_DIM)
    for i in range(sys.maxunicode): 
        try:
            embedding[chr(i)] = np.random.uniform(-1,0,EMBEDDING_DIM)
        except:
            continue

    #embedding = dict()
    #for line in embed_file:
    #    values = line.split()
    #    values.append(1.0)
    #    embedding[values[0]] = np.array(values[1:]).astype(np.float)
    return embedding

def parse_file(train_filename, embedding, use_max_len=True):
    print("Parsing UD file...")
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

    print("Sentences:",sentences[:10])
    print("Labels:",labels[:10])
    return sentences, labels

def parse_morph_langid_file(train_filename, embedding, use_max_len=True):
    print("Parsing morphology/langid file...")
    train_file = open(train_filename)
    sentences = []
    labels = []

    for line in train_file:
        if not line:
            continue

        line = line.split('\t')

        tags = line[2].split()
        segs = line[1].split()

        if len(tags) != len(segs):
            print(tags)
            print(segs)
            continue

        labs = list()

        for tag, seg in zip(tags,segs):
            labs.append((tag.rstrip(),len(seg)))


        labels.append((labs, line[0]))

        N = len(line[0])
        if use_max_len:
            max_len = MAX_SENTENCE_LEN
        else:
            max_len = N
        sentence_vec = np.zeros((max_len, EMBEDDING_DIM))
        for i in range(min(N, max_len)):
            c = line[0][i]
            if c in embedding:
                sentence_vec[i, :] = embedding[c]
            elif c in "0123456789":
                sentence_vec[i, :] = embedding["<NUM>"]
            else:
                sentence_vec[i, :] = embedding["<unk>"]
        sentences.append(sentence_vec)

    print("Sentences:",sentences[:10])
    print("Labels:",labels[:10])
 
    return sentences, labels


