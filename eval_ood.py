# -*- coding: utf-8 -*-
"""
Evaluate a Intent classifier with OE.
"""

import numpy as np
import sys
import os
import pickle
import argparse
import math
import time
import pandas as pd
from bisect import bisect_left
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchtext

from torchtext import data
from torchtext import datasets

import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

np.random.seed(1)

parser = argparse.ArgumentParser(description='IC OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dist_dataset', type=str, default='v1.0')
parser.add_argument('--oe_dataset', type=str, default='v1.0')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
args = parser.parse_args()

torch.set_grad_enabled(False)
cudnn.benchmark = True

if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import get_performance


TEXT = data.Field(pad_first=True, lower=True, fix_length=100)
LABEL = data.Field(sequential=False)

train_path = './data/' + args.in_dist_dataset + '/train.csv'
test_path =  './data/' + args.in_dist_dataset + '/test_1.csv'

train = data.TabularDataset(path= train_path, format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

# test = data.TabularDataset(path=test_path, format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)
test = data.TabularDataset(path=test_path, format='csv', fields=[('text', TEXT)], skip_header=True)

TEXT.build_vocab(train, max_size=10000)
LABEL.build_vocab(train, max_size=10000)
print('Vocab length (including special tokens):', len(TEXT.vocab))
print("\nList label : {}\n".format(LABEL.vocab.stoi.keys()))



vocab_dict = TEXT.vocab.stoi
txt2id = {}
for key, value in vocab_dict.items():
    txt2id[value] = key

def convert(idx_input):
    a = [txt2id[d] for d in idx_input if d != 1]
    return " ".join(a)


train_iter = data.BucketIterator(train, batch_size=args.batch_size, repeat=False)
test_iter = data.BucketIterator(test, batch_size=1, repeat=False)

# ood_num_examples = len(test_iter.dataset) // 5
ood_num_examples = len(test_iter.dataset)
expected_ap = ood_num_examples / (ood_num_examples + len(test_iter.dataset))
recall_level = 0.9
# recall_level = 0

# ================ OE dataset ================ #
TEXT_oe = data.Field(pad_first=True, lower=True)
oe_data = data.TabularDataset(path='./data/{}/oe_test.csv'.format(args.oe_dataset), 
                                format='csv', 
                                fields=[('text', TEXT_oe)],
                                skip_header=True)

TEXT_oe.build_vocab(train.text, max_size=10000)
print('Vocab length (including special tokens): ', len(TEXT_oe.vocab))

train_iter_oe = data.BucketIterator(oe_data, batch_size=args.batch_size, repeat=False)
# ================ OE dataset ================ #

class ClfGRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), 50, padding_idx=1)
        self.gru = nn.GRU(input_size=50, hidden_size=128, num_layers=2, bias=True, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        embeds = self.embedding(x)
        hidden = self.gru(embeds)[1][1] # select h_n and select the 2nd layer
        logits = self.linear(hidden)
        return logits

model = ClfGRU(6).cuda()
model.load_state_dict(torch.load('./models/v1.0/OE/model_finetune.dict'))
print('\nLoaded model.\n')
import numpy as np
import sklearn.metrics as sk
from sklearn.preprocessing import binarize

# def get_scores(dataset_iterator, ood=False):
#     model.eval()

#     outlier_scores = []
#     list_text = []
#     # print(dataset_iterator.dataset[0].text)

#     for batch_idx, batch in enumerate(iter(dataset_iterator)):
#         if ood and (batch_idx * args.batch_size > ood_num_examples):
#             break

#         inputs = batch.text.t()
#         # print(batch_idx)
#         # print(inputs)
#         inputs = inputs.to(device)
#         logits = model(inputs)
#         smax = F.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
#         msp = -1 * torch.max(smax, dim=1)[0]
#         # ce_to_unif = F.log_softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1).mean(1)
#         outlier_scores.extend(list(msp.data.cpu().numpy()))
#         # print(outlier_scores)

#         temp = inputs.cpu().numpy().tolist()
#         # print(type(temp)
#         for txt in temp:
#             # print(convert(txt))
#             list_text.append(convert(txt))
#         # print(list_text)
#             # print(convert(txt))

#         # text = convert(inputs.cpu().numpy()[0].tolist)
#     temp_outlier_scores = np.array(outlier_scores)
#     temp_outlier_scores = temp_outlier_scores.reshape(1, -1)
#     list_score = binarize(temp_outlier_scores, -0.5)[0]
#     df = pd.DataFrame({
#         'Sentence': list_text,
#         'Score'   : list_score,
#         # 'Label'   : 
#     })
#     df.to_csv('df_fail.csv', encoding='utf-8-sig', index=False)
#     return outlier_scores

# test_scores = get_scores(test_iter)

from torch.autograd import Variable


df_test = pd.read_csv("data/v1.0/test_1.csv", encoding='utf-8')
list_text = df_test.sentence.values.tolist()
list_score = []
# print(list_text)
text = "Thanks c nhìu nha"
for text in list_text:
    test_sen = TEXT.preprocess(text)
    test_sen = [[TEXT.vocab.stoi[x] for x in test_sen]]
    test_sen = np.asarray(test_sen)
    test_sen = torch.LongTensor(test_sen)
    test_tensor = Variable(test_sen, requires_grad=False)
    test_tensor = test_tensor.cuda()

    model.eval()
    logits = model(test_tensor)
    smax = F.softmax(logits - torch.max(logits, dim=1, keepdim=True)[0], dim=1)
    msp = -1 * torch.max(smax, dim=1)[0]
    a = binarize(msp.data.cpu().numpy().reshape(1, -1), -0.5)[0].item(0)
    list_score.append(a)

df_final = pd.DataFrame({
    'Sentence' : list_text,
    'Score'    : list_score
})
df_final.to_csv('df_fail.csv', encoding='utf-8-sig', index=False)

# mean_fprs = []
# mean_aurocs = []
# mean_auprs = []

# title = 'oe'
# iterator = train_iter_oe

# print('\n{}'.format(title))
# fprs, aurocs, auprs = [], [], []
# for i in range(10):
#     ood_scores = get_scores(iterator, ood=True)
#     fpr, auroc, aupr = get_performance(ood_scores, test_scores, expected_ap, recall_level=recall_level)
#     fprs.append(fpr)
#     aurocs.append(auroc)
#     auprs.append(aupr)

# print("\n")
# print('FPR{:d}:\t\t\t{:.4f} ({:.4f})'.format(int(100 * recall_level), np.mean(fprs), np.std(fprs)))
# print('AUROC:\t\t\t{:.4f} ({:.4f})'.format(np.mean(aurocs), np.std(aurocs)))
# print('AUPR:\t\t\t{:.4f} ({:.4f})'.format(np.mean(auprs), np.std(auprs)))
