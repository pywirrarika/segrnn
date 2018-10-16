#!/bin/bash

mkdir -p models

python seg_rnn.py --train ../data/tr/train.1.tr --test ../data/tr/dev.1.tr --morph  --modelPath models # --model seg_rnn_correct_0.pt
