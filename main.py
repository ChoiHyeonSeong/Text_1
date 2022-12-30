import os
import math
import random
import pandas as pd
import regex as re
import numpy as np
from typing import Optional, Sequence

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score

from tqdm import tqdm
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, AutoModel, AutoConfig

import gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')



'''
기본 뼈대:
class CustomDataset(torch.utils.data.Dataset): 
  def __init__(self):

  def __len__(self):

  def __getitem__(self, idx):

메서드 설명:
__init__(self) : 필요한 변수들을 선언하는 메서드. input으로 오는 x와 y를 load 하거나, 파일목록을 load한다.
__len__(self) : x나 y 는 길이를 넘겨주는 메서드.
__getitem__(self, index) : index번째 데이터를 return 하는 메서드.
'''
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            st_type = self.labels['type'][idx]
            st_polarity = self.labels['polarity'][idx]
            st_tense = self.labels['tense'][idx]
            st_certainty = self.labels['certainty'][idx]
            item["labels"] = torch.tensor(st_type), torch.tensor(st_polarity), torch.tensor(st_tense), torch.tensor(
                st_certainty)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

'''
'''
# Define trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        labels = inputs.pop("labels").to(torch.int64)

        type_logit, polarity_logit, tense_logit, certainty_logit = model(**inputs)

        # # simple loss
        # criterion = {
        #     'type' : nn.CrossEntropyLoss().to(device),
        #     'polarity' : nn.CrossEntropyLoss().to(device),
        #     'tense' : nn.CrossEntropyLoss().to(device),
        #     'certainty' : nn.CrossEntropyLoss().to(device)
        # }
        # loss = criterion['type'](type_logit, labels[::, 0]) + \
        #             criterion['polarity'](polarity_logit, labels[::, 1]) + \
        #             criterion['tense'](tense_logit,labels[::, 2]) + \
        #             criterion['certainty'](certainty_logit, labels[::, 3])

        # focal loss
        criterion = {
            'type': FocalLoss().to(device),
            'polarity': FocalLoss().to(device),
            'tense': FocalLoss().to(device),
            'certainty': FocalLoss().to(device)
        }
        # labels = labels.type(torch.float).clone().detach()
        loss = criterion['type'](type_logit, labels[::, 0]) + \
               criterion['polarity'](polarity_logit, labels[::, 1]) + \
               criterion['tense'](tense_logit, labels[::, 2]) + \
               criterion['certainty'](certainty_logit, labels[::, 3])

        outputs = None, \
                  torch.argmax(type_logit, dim=1), \
                  torch.argmax(polarity_logit, dim=1), \
                  torch.argmax(tense_logit, dim=1), \
                  torch.argmax(certainty_logit, dim=1)
        return (loss, outputs) if return_outputs else loss