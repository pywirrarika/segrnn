import argparse
import random
import time
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from config import *
from preproc import parse_embedding, parse_embedding_fake, parse_file, parse_morph_langid_file
from evaluate import eval_f1
from model import SegRNN

def count_correct_labels(predicted, gold):
    correct_count = 0
    predicted_set = set()
    chars = 0
    for tag, l in predicted:
        label = (tag, chars, chars + l)
        predicted_set.add(label)
        chars += l
    chars = 0
    for tag, l in gold:
        label = (tag, chars, chars + l)
        if label in predicted_set:
            correct_count += 1
        chars += l
    return correct_count

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmental RNN.')
    parser.add_argument('--train', help='Training file')
    parser.add_argument('--test', help='Test file')
    parser.add_argument('--embed', help='Character embedding file')
    parser.add_argument('--model', help='Saved model')
    parser.add_argument('--lr', help='Learning rate (default=0.01)')
    parser.add_argument('--evalModel', help='Evaluate this model')
    parser.add_argument('--morph', action='store_true')

    args = parser.parse_args()

    if args.embed:
        embedding = parse_embedding(args.embed)
        print("Done parsing embedding")
    else:
        embedding = parse_embedding_fake(args.embed)
        print("Training without char embeddings")


    if args.test is not None:
        if args.morph:
            test_data, test_labels = parse_morph_langid_file(args.test, embedding, False)
        else:
            test_data, test_labels = parse_file(args.test, embedding, False)
        test_pairs = list(zip(test_data, test_labels))
        print("Done parsing testing data")

    if args.evalModel is not None:
        eval_rnn = torch.load(args.evalModel)
        eval_f1(eval_rnn, test_pairs, False)
        import sys
        sys.exit(0)

    if args.morph:
        data, labels = parse_morph_langid_file(args.train, embedding, use_max_sentence_len_training)
    else:
        data, labels = parse_file(args.train, embedding, use_max_sentence_len_training)

    pairs = list(zip(data, labels))
    # pairs = pairs[0:250]
    print("Done parsing training data")

    if args.model is not None:
        seg_rnn = torch.load(args.model)
    else:
        seg_rnn = SegRNN()

    if args.lr is not None:
        learning_rate = float(args.lr)
    else:
        learning_rate = 0.01

    optimizer = torch.optim.Adam(seg_rnn.parameters(), lr=learning_rate)
    count = 0.0
    sum_loss = 0.0
    correct_count = 0.0
    sum_gold = 0.0
    sum_pred = 0.0
    for batch_num in range(1000):
        random.shuffle(pairs)
        if use_bucket_training:
            bucket_pairs = pairs[0:BATCH_SIZE]
            bucket_pairs.sort(key=lambda x:x[0].shape[0])
        else:
            bucket_pairs = pairs
        for i in range(0, min(BATCH_SIZE, len(pairs)), MINIBATCH_SIZE):
            seg_rnn.train()
            start_time = time.time()

            optimizer.zero_grad()
            
            if use_bucket_training:
                batch_size = min(MINIBATCH_SIZE, len(pairs) - i)
                max_len = bucket_pairs[i][0].shape[0]
                print(bucket_pairs[i][0].shape[0])
                print(bucket_pairs[i + batch_size - 1][0].shape[0])
            elif use_max_sentence_len_training:
                max_len = MAX_SENTENCE_LEN
                batch_size = min(MINIBATCH_SIZE, len(pairs) - i)
            else:
                max_len = len(pairs[i][1][1])
                batch_size = 1
            batch_data = np.zeros((max_len, batch_size, EMBEDDING_DIM))
            batch_labels = []
            for idx, (datum, (label, sentence)) in enumerate(bucket_pairs[i:i+batch_size]):
                batch_data[:, idx, :] = datum[0:max_len, :]
                batch_labels.append(label)
            loss = seg_rnn.calc_loss(batch_data, batch_labels)
            print("LOSS:", loss.item())
            sum_loss = loss.item()
            count = 1.0 * batch_size
            loss.backward()

            optimizer.step()

            seg_rnn.eval()
            print("Batch ", batch_num, " datapoint ", i, " avg loss ", sum_loss / count)
            if i % 16 == 0:
                sentence_len = len(bucket_pairs[i][1][1])
                pred = seg_rnn.infer(batch_data[0:sentence_len, 0, np.newaxis, :])
                gold = bucket_pairs[i][1][0]
                print('Prediction:')
                print(pred)
                print('Gold:')
                print(gold)
                print(bucket_pairs[i][1][1], sentence_len)
                sentence_unk = ""
                for c in bucket_pairs[i][1][1]:
                    sentence_unk += c if c in embedding or c in "0123456789" else "_"
                print(sentence_unk)
                correct_count += count_correct_labels(pred, gold)
                sum_gold += len(gold)
                sum_pred += len(pred)
                cum_prec = correct_count / sum_pred
                cum_rec = correct_count / sum_gold
                if cum_prec > 0 and cum_rec > 0:
                    print("F1: ", 2.0 / (1.0 / cum_prec + 1.0 / cum_rec)," cum. precision: ", cum_prec, " cum. recall: ", cum_rec)
                # print(seg_rnn.Y_encoding[0], seg_rnn.Y_encoding[5])
                # print(seg_rnn.Y_encoding[0].grad, seg_rnn.Y_encoding[5].grad)
                #for param in seg_rnn.parameters():
                #    print(param)

            end_time = time.time()
            print("Took ", end_time - start_time, " to run ", MINIBATCH_SIZE, " training sentences")

        if args.test is not None:
            torch.save(seg_rnn, "seg_rnn_correct_" + str(batch_num) + ".pt")
            #if (batch_num + 1) % 40 == 0:
            #    eval_f1(seg_rnn, test_pairs)
