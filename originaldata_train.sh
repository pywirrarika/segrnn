#!/bin/bash

mkdir -p models

#python seg_rnn.py --train data/UD_French-ParTUT/fr_partut-ud-train.conllu --test data/UD_French-ParTUT/fr_partut-ud-test.conllu --modelPath models # --model seg_rnn_correct_0.pt
python seg_rnn.py --train data/UD_Chinese-GSD/zh_gsd-ud-train.conllu --test data/UD_Chinese-GSD/zh_gsd-ud-dev.conllu --modelPath models --embed etc/wiki.zh.vec # --model seg_rnn_correct_0.pt
