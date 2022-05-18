#-*- encoding: utf-8 -*-
import sys
import io

from zmq import device
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

from argparse import Namespace
from collections import Counter
import json
import os
import string

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm

from Summary.surnmame_classifier import *

args = Namespace(
    #날짜 및 경로
    surname_csv = 'chapter_4/4_2_mlp_surnames/data/surnames/surnames_with_splits.csv',
    vectorizer_file='vectorizer.json',
    model_state_file='model.path',
    save_dir='chapter_4/4_2_mlp_surnames/model_storage/ch4/surname_mlp',
    # 모델 파라미터
    hidden_dim=300,
    # 훈련 파라미터
    seed=1998,
    num_epochs=100,
    early_stopping_criteria=5, 
    learning_rate=0.001,
    batch_size=64,
    device='cuda'
)


'''
데이터셋, 모델, loss, optimizer
'''
dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv) # 입력: 경로
vectorizer = dataset.get_vectorizer()
classifier = SurnameClassifier(input_dim=len(vectorizer.surname_vocab), hidden_dim=args.hidden_dim, output_dim=len(vectorizer.nationality_vocab))
classifier.to(args.device)

loss_func = nn.CrossEntropyLoss(dataset.class_weights)
optimizer = optim.AdamW(classifier.parameters(), lr=args.learning_rate)
'''
훈련반복
'''
optimizer.zero_grad()

y_pred = classifier(batch_di)