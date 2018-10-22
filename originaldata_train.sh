#!/bin/bash

mkdir -p models

python seg_rnn.py --train data/UD_French-ParTUT/fr_partut-ud-train.conllu --test data/UD_French-ParTUT/fr_partut-ud-test.conllu --modelPath models # --model seg_rnn_correct_0.pt
