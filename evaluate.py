from config import *

def eval_f1(seg_rnn, pairs, write_to_file=True):
    gold_segs = 0
    predicted_segs = 0
    correct_segs = 0
    for idx, (datum, (gold_label, sentence)) in enumerate(pairs):
        if idx % 25 == 0:
            print("eval ", idx)
        predicted_label = seg_rnn.infer(datum.reshape(len(sentence), 1, EMBEDDING_DIM))
        predicted_segs += len(predicted_label)
        gold_segs += len(gold_label)
        correct_segs += count_correct_labels(predicted_label, gold_label)
    if predicted_segs > 0:
        precision = correct_segs / predicted_segs
    else:
        precision = 0.0
    print("Precision: ", precision)
    if gold_segs > 0:
        recall = correct_segs / gold_segs
    else:
        recall = 0.0
    print("Recall: ", recall)
    if precision > 0 and recall > 0:
        f1 = 2.0 / (1.0 / precision + 1.0 / recall)
    else:
        f1 = 0.0
    print("F1: " , f1)
    if write_to_file:
        f = open("eval_scores.txt", "a+")
        f.write("%f %f %f\n" % (precision, recall, f1))
        f.close()

