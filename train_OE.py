# -*- coding: utf-8 -*-
"""
Trains a Intent classifier with Outlier Exposure.
"""

import numpy as np
import sys
import os
import pickle
import argparse
import math
import time
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

parser = argparse.ArgumentParser(description='Train with OE',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dist_dataset', type=str, default='v1.0')
parser.add_argument('--oe_dataset', type=str, default='v1.0')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.5, help='Momentum.')
parser.add_argument('--test_bs', type=int, default=256)
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./models', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='./models', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
parser.add_argument('--mix', dest='mix', action='store_true', help='Mix outliers sentences with in-dist sentences.')
# Acceleration
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
args = parser.parse_args()


TEXT = data.Field(pad_first=True, lower=True, fix_length=100)
LABEL = data.Field(sequential=False)

train_path = './data/' + args.in_dist_dataset + '/train.csv'
test_path =  './data/' + args.in_dist_dataset + '/test.csv'

train = data.TabularDataset(path= train_path, format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)

test = data.TabularDataset(path=test_path, format='csv', fields=[('text', TEXT), ('label', LABEL)], skip_header=True)


TEXT.build_vocab(train, max_size=10000)
LABEL.build_vocab(train, max_size=10000)
print('Vocab length (including special tokens):', len(TEXT.vocab))
print("\nList label : {}\n".format(LABEL.vocab.stoi.keys()))

train_iter = data.BucketIterator(train, batch_size=args.batch_size, repeat=False)
test_iter = data.BucketIterator(test, batch_size=args.batch_size, repeat=False)

### Outlier Exposure Dataset
TEXT_custom = data.Field(pad_first=True, lower=True)
oe_path = './data/' + args.oe_dataset + '/oe_train.csv'
custom_data = data.TabularDataset(path=oe_path, format='csv', fields=[('text', TEXT_custom)], skip_header=True)

TEXT_custom.build_vocab(train.text, max_size=10000)
print('Vocab length (including special tokens):', len(TEXT_custom.vocab))

train_iter_oe = data.BucketIterator(custom_data, batch_size=args.batch_size, repeat=False)


cudnn.benchmark = True  # fire on all cylinders


class ClfGRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), 50, padding_idx=1)
        self.gru = nn.GRU(input_size=50, hidden_size=128, num_layers=2,
            bias=True, batch_first=True,bidirectional=False)
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        embeds = self.embedding(x)
        hidden = self.gru(embeds)[1][1]  # select h_n, and select the 2nd layer
        logits = self.linear(hidden)
        return logits


model = ClfGRU(6).cuda()  # change to match dataset
model.load_state_dict(torch.load('./models/{}/baseline/model.dict'.format(args.oe_dataset)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


def train():
    model.train()
    data_loss_ema = 0
    oe_loss_ema = 0

    for batch_idx, (batch, batch_oe) in enumerate(zip(iter(train_iter), iter(train_iter_oe))):
        inputs = batch.text.t()
        labels = batch.label - 1
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        logits = model(inputs)
        data_loss = F.cross_entropy(logits, labels)

        inputs_oe = batch_oe.text.t()
        inputs_oe = inputs_oe.to(device)
        logits_oe = model(inputs_oe)
        smax_oe = F.log_softmax(logits_oe - torch.max(logits_oe, dim=1, keepdim=True)[0], dim=1)
        oe_loss = -1 * smax_oe.mean()  # minimizing cross entropy

        loss = data_loss + oe_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        data_loss_ema = data_loss_ema * 0.9 + data_loss.data.cpu().numpy() * 0.1
        oe_loss_ema = oe_loss_ema * 0.9 + oe_loss.data.cpu().numpy() * 0.1

        if (batch_idx % 200 == 0 or batch_idx < 10):
            print('iter: {} \t| data_loss_ema: {} \t| oe_loss_ema: {}'.format(
                batch_idx, data_loss_ema, oe_loss_ema))

    scheduler.step()

from sklearn.metrics import recall_score, f1_score, precision_score

def evaluate():
    model.eval()
    running_loss = 0
    num_examples = 0
    correct = 0

    pres_list  = []
    label_list = []

    for batch_idx, batch in enumerate(iter(test_iter)):
        inputs = batch.text.t()
        labels = batch.label - 1
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)

        loss = F.cross_entropy(logits, labels, size_average=False)
        running_loss += loss.data.cpu().numpy()

        pred = logits.max(1)[1]
        correct += pred.eq(labels).sum().data.cpu().numpy()

        num_examples += inputs.shape[0]
        pres_list.extend(pred.cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    acc = correct / num_examples
    loss = running_loss / num_examples

    print("Recall-Score")
    print(recall_score(label_list, pres_list, average='macro') )

    print("F1-Score")
    print(f1_score(label_list, pres_list, average='macro'))

    print("Precision-Score")
    print(precision_score(label_list, pres_list, average='macro'))
    print(pres_list)
    return acc, loss


acc, loss = evaluate()
print('Test acc: {} \t| Test loss: {}\n'.format(acc, loss))
for epoch in range(args.epochs):
    print('Epoch', epoch)
    train()
    acc, loss = evaluate()
    print('Test acc: {} \t| Test loss: {}\n'.format(acc, loss))

if not os.path.exists('./models/{}/OE/'.format(args.in_dist_dataset)):
    os.makedirs('./models/{}/OE/'.format(args.in_dist_dataset))
torch.save(model.state_dict(), './models/{}/OE/model_finetune.dict'.format(args.in_dist_dataset))
print('Saved model.')
