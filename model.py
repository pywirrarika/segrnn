import random
import time
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from config import *
from preproc import parse_embedding, parse_embedding_fake, parse_file
from evaluate import eval_f1

def logsumexp(inputs, dim=None, keepdim=False):
        return (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)

# SegRNN module
class SegRNN(nn.Module):
    def __init__(self):
        super(SegRNN, self).__init__()
        self.forward_context_initial = (nn.Parameter(torch.randn(LAYERS_1, 1, XCRIBE_DIM)), nn.Parameter(torch.randn(LAYERS_1, 1, XCRIBE_DIM)))
        self.backward_context_initial = (nn.Parameter(torch.randn(LAYERS_1, 1, XCRIBE_DIM)), nn.Parameter(torch.randn(LAYERS_1, 1, XCRIBE_DIM)))
        self.forward_context_lstm = nn.LSTM(INPUT_DIM, XCRIBE_DIM, LAYERS_1, dropout=DROPOUT)
        self.backward_context_lstm = nn.LSTM(INPUT_DIM, XCRIBE_DIM, LAYERS_1, dropout=DROPOUT)
        self.register_parameter("forward_context_initial_0", self.forward_context_initial[0])
        self.register_parameter("forward_context_initial_1", self.forward_context_initial[1])
        self.register_parameter("backward_context_initial_0", self.backward_context_initial[0])
        self.register_parameter("backward_context_initial_1", self.backward_context_initial[1])

        self.forward_initial = (nn.Parameter(torch.randn(LAYERS_2, 1, SEG_DIM)), nn.Parameter(torch.randn(LAYERS_2, 1, SEG_DIM)))
        self.backward_initial = (nn.Parameter(torch.randn(LAYERS_2, 1, SEG_DIM)), nn.Parameter(torch.randn(LAYERS_2, 1, SEG_DIM)))
        self.Y_encoding = [nn.Parameter(torch.randn(1, 1, TAG_DIM)) for i in range(len(LABELS))]
        self.Z_encoding = [nn.Parameter(torch.randn(1, 1, DURATION_DIM)) for i in range(1, DATA_MAX_SEG_LEN + 1)]

        self.register_parameter("forward_initial_0", self.forward_initial[0])
        self.register_parameter("forward_initial_1", self.forward_initial[1])
        self.register_parameter("backward_initial_0", self.backward_initial[0])
        self.register_parameter("backward_initial_1", self.backward_initial[1])
        for idx, encoding in enumerate(self.Y_encoding):
            self.register_parameter("Y_encoding_" + str(idx), encoding)
        for idx, encoding in enumerate(self.Z_encoding):
            self.register_parameter("Z_encoding_" + str(idx), encoding)

        self.forward_lstm = nn.LSTM(2 * XCRIBE_DIM, SEG_DIM, LAYERS_2)
        self.backward_lstm = nn.LSTM(2 * XCRIBE_DIM, SEG_DIM, LAYERS_2)
        self.V = nn.Linear(SEG_DIM + SEG_DIM + TAG_DIM + DURATION_DIM, SEG_DIM)
        self.W = nn.Linear(SEG_DIM, 1)
        self.Phi = nn.Tanh()

    def calc_loss(self, batch_data, batch_label):
        N, B, K = batch_data.shape
        #print(B, len(batch_label))
        #print(N, B, K)
        forward_precalc, backward_precalc = self._precalc(batch_data)

        log_alphas = [autograd.Variable(torch.zeros((1, B, 1)))]
        for i in range(1, N + 1):
            t_sum = []
            for j in range(max(0, i - DATA_MAX_SEG_LEN), i):
                precalc_expand = torch.cat([forward_precalc[j][i - 1], backward_precalc[j][i - 1]], 2).repeat(len(LABELS), 1, 1)
                y_encoding_expand = torch.cat([self.Y_encoding[y] for y in range(len(LABELS))], 0).repeat(1, B, 1)
                z_encoding_expand = torch.cat([self.Z_encoding[i - j - 1] for y in range(len(LABELS))]).repeat(1, B, 1)
                # LABELS, MINIBATCH, FEATURES
                seg_encoding = torch.cat([precalc_expand, y_encoding_expand, z_encoding_expand], 2)
                # Linear thru features: LABELS, MINIBATCH, 1
                t = self.W(self.Phi(self.V(seg_encoding)))
                # summed across labels: 1, MINIBATCH, 1
                summed_t = logsumexp(t, 0, True)
                t_sum.append(log_alphas[j] + summed_t)
            # cat across seglenths: SEG_LENGTH, MINIBATCH, 1
            all_t_sums = torch.cat(t_sum, 0)
            # sum across lengths: 1, MINIBATCH, 1
            new_log_alpha = logsumexp(all_t_sums, 0, True)
            log_alphas.append(new_log_alpha)

        loss = torch.sum(log_alphas[N])

        for batch_idx in range(B):
            indiv = autograd.Variable(torch.zeros(1))
            chars = 0
            label = batch_label[batch_idx]
            for tag, length in label:
                if length > DATA_MAX_SEG_LEN:
                    chars += length
                    continue
                if chars + length > N:
                    break
                forward_val = forward_precalc[chars][chars + length - 1][:, batch_idx, np.newaxis, :]
                backward_val = backward_precalc[chars][chars + length - 1][:, batch_idx, np.newaxis, :]
                try:
                    y_val = self.Y_encoding[LABELS.index(tag)]
                except:
                    print("Error:",tag)
                    y_val = self.Y_encoding[LABELS.index('UNK')]
                z_val = self.Z_encoding[length - 1]
                seg_encoding = torch.cat([forward_val, backward_val, y_val, z_val], 2)
                #print(seg_encoding.size)
                indiv += self.W(self.Phi(self.V(seg_encoding)))[0][0]
                chars += length
            loss -= indiv[0]
        return loss

    def _precalc(self, data):
        N, B, K = data.shape

        forward_xcribe_data = []
        hidden = (
            torch.cat([self.forward_context_initial[0] for b in range(B)], 1),
            torch.cat([self.forward_context_initial[1] for b in range(B)], 1)
        )
        for i in range(N):
            next_input = autograd.Variable(torch.from_numpy(data[i, :]).float())
            out, hidden = self.forward_context_lstm(next_input.view(1, B, K), hidden)
            forward_xcribe_data.append(out)
        backward_xcribe_data = []
        hidden = (
            torch.cat([self.backward_context_initial[0] for b in range(B)], 1),
            torch.cat([self.backward_context_initial[1] for b in range(B)], 1)
        )
        for i in range(N - 1, -1, -1):
            next_input = autograd.Variable(torch.from_numpy(data[i, :]).float())
            out, hidden = self.backward_context_lstm(next_input.view(1, B, K), hidden)
            backward_xcribe_data.append(out)

        backward_xcribe_data.reverse()

        xcribe_data = []
        for i in range(N):
            xcribe_data.append(torch.cat([forward_xcribe_data[i], backward_xcribe_data[i]], 2))

        forward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # forward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = (
                torch.cat([self.forward_initial[0] for b in range(B)], 1),
                torch.cat([self.forward_initial[1] for b in range(B)], 1)
            )
            for j in range(i, min(N, i + DATA_MAX_SEG_LEN)):
                next_input = xcribe_data[j]
                out, hidden = self.forward_lstm(next_input, hidden)
                forward_precalc[i][j] = out

        backward_precalc = [[None for _ in range(N)] for _ in range(N)]
        # backward_precalc[i, j, :] => [i, j]
        for i in range(N):
            hidden = (
                torch.cat([self.backward_initial[0] for b in range(B)], 1),
                torch.cat([self.backward_initial[1] for b in range(B)], 1)
            )
            for j in range(i, max(-1, i - DATA_MAX_SEG_LEN), -1):
                next_input = xcribe_data[j]
                out, hidden = self.backward_lstm(next_input, hidden)
                backward_precalc[j][i] = out
        return forward_precalc, backward_precalc

    def infer(self, data):
        N, B, K = data.shape
        forward_precalc, backward_precalc = self._precalc(data)
        
        log_alphas = [(-1, -1, 0.0)]
        for i in range(1, N + 1):
            t_sum = []
            max_len = -1
            max_t = float("-inf")
            max_label = -1
            for j in range(max(0, i - DATA_MAX_SEG_LEN), i):
                precalc_expand = torch.cat([forward_precalc[j][i - 1], backward_precalc[j][i - 1]], 2).repeat(len(LABELS), 1, 1)
                y_encoding_expand = torch.cat([self.Y_encoding[y] for y in range(len(LABELS))], 0)
                z_encoding_expand = torch.cat([self.Z_encoding[i - j - 1] for y in range(len(LABELS))])
                seg_encoding = torch.cat([precalc_expand, y_encoding_expand, z_encoding_expand], 2)
                t_val = self.W(self.Phi(self.V(seg_encoding)))
                t = t_val + log_alphas[j][2]
                # print("t_val: ", t_val)
                for y in range(len(LABELS)):
                    if t.data[y, 0, 0] > max_t:
                        max_t = t.data[y, 0, 0]
                        max_label = y
                        max_len = i - j
            log_alphas.append((max_label, max_len, max_t))

        cur_pos = N
        ret = []
        while cur_pos != 0:
            ret.append((LABELS[log_alphas[cur_pos][0]], log_alphas[cur_pos][1]))
            cur_pos -= log_alphas[cur_pos][1]
        return list(reversed(ret))


