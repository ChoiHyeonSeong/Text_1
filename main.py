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