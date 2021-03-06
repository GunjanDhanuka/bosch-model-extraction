# Importing the required libraries
import os
import secrets
import pandas as pd
import string
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import os.path as osp
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.apis import train_model
import mmcv
from mmcv import Config

# TODO: Make changes according to greybox and add the config of greybox
val = pd.read_csv("P1/data/validation400.csv")
train = pd.read_csv("P1/data/combined400.csv")

# Since the dataset was prepared on kaggle, we need to replace it to get absolute path.
train['path'] = train['path'].str.replace('/kaggle/input/', '')

for i, pth in enumerate(train['path']):
    if not os.path.exists(pth):
        train.loc[i, 'path'] = "animalsvids" + train.loc[i, 'path']
train = train[:-1]


# Loading the list of labels of the Kinetics-400 dataset
label = 'P1/data/label_map_k400.txt'
labels = open(label).readlines()
labels = [l.strip() for l in labels]
label_to_id = {l: i for i, l in enumerate(labels)}

train['label_id'] = train['label'].map(label_to_id)
val['label_id'] = val['label'].map(label_to_id)

# TRAIN ANNOTATIONS
texts = []
file = open("P1/data/train_annotations.txt", "w+")

for _, row in train.iterrows():
    name = row.path[1:]
    text = f"{name} {row.label_id} \n"
    texts.append(text)

file.writelines(texts)
file.close()

# VAL ANNOTATIONS
texts = []
file = open("P1/data/val_annotations.txt", "w+")
for _, row in val.iterrows():
    name = row.path[1:]
    text = f"{name} {row.label_id} \n"
    texts.append(text)

file.writelines(texts)
file.close()

# Loading configuration for training the model
from mmcv import Config
cfg = Config.fromfile('config/greybox_config.py')
print(f'Config:\n{cfg.pretty_text}')
# Using MMAction utilities for building the dataset and training the model.

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the recognizer
model = build_model(cfg.model, train_cfg=cfg.get(
    'train_cfg'), test_cfg=cfg.get('test_cfg'))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

# Train the model
train_model(model, datasets, cfg, distributed=False, validate=True)