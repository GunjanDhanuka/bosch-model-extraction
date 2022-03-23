import os
import gc
import copy
import time
import random
import glob
import string
import wandb
import shutil

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Utils
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, train_test_split

# PytorchVideo Imports
import pytorchvideo
from pytorchvideo.transforms import (
    UniformCropVideo,
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)
from pytorchvideo.models import create_res_basic_head

# # For colored terminal text
# from colorama import Fore, Back, Style
#
# b_ = Fore.BLUE
# y_ = Fore.YELLOW
# sr_ = Style.RESET_ALL
#
# # Suppress warnings
# import warnings
#
# warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

Config = {"seed": 2022,
          "data_path": "/content/dataset",
          "epochs": 15,
          "size": 768,
          "train_batch_size": 22,
          "valid_batch_size": 22,
          "num_workers": 12,
          "lr": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_0": 20,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "num_labels": 600,
          "margin": 0.5,
          "clip_duration": 2,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          'wandb': True,
          'apex': True,
          'n_accum': 1,
          "debug": False
          }


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(Config['seed'])

# TODO: Fix the "files" path here
files = glob.glob("/content/data/**/**/*.mp4")
labels = [file.split("/")[-2] for file in files]
print(f"No of videos: {len(files)}")

df = pd.DataFrame({"file_path": files, "labels": labels})
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['labels'])

# TODO: Local dataset creation to be kept or removed?
for label in np.unique(labels):
    os.makedirs(f"dataset/train/{label}", exist_ok=True)
    os.makedirs(f"dataset/val/{label}", exist_ok=True)

for _, row in train_df.iterrows():
    label = row.file_path.split("/")[-2]
    file_name = row.file_path.split("/")[-1]
    shutil.copy(row["file_path"], f"dataset/train/{label}/{file_name}")

for _, row in val_df.iterrows():
    label = row.file_path.split("/")[-2]
    file_name = row.file_path.split("/")[-1]
    shutil.copy(row["file_path"], f"dataset/val/{label}/{file_name}")

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway

        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )

        frame_list = [slow_pathway, fast_pathway]
        return frame_list


# TODO: Can explain these transformations either in code/ppt/doc
train_transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    )
)

# The duration of the input clip is also specific to the model.
Config['clip_duration'] = (num_frames * sampling_rate) / frames_per_second

train_dataset = pytorchvideo.data.Kinetics(
    data_path=os.path.join(Config['data_path'], "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", Config["clip_duration"]),
    transform=train_transform,
    decode_audio=False
)

val_dataset = pytorchvideo.data.Kinetics(
    data_path=os.path.join(Config["data_path"], "val"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("random", Config["clip_duration"]),
    transform=train_transform,
    decode_audio=False
)

# Creating dataloaders using PyTorch Dataloader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=Config["train_batch_size"],
    num_workers=Config["num_workers"],
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=Config["valid_batch_size"],
    num_workers=Config["num_workers"],
)


class videoModel(nn.Module):
    def __init__(self):
        super(videoModel, self).__init__()
        model_name = "slowfast_r50"
        self.model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            model=model_name,
            pretrained=False
        )
        self.model.blocks[-1] = create_res_basic_head(
            in_features=2304,
            out_features=Config["num_labels"],
            pool=None
        )

    def forward(self, x):
        output = self.model(x)
        return output


model = videoModel()
model.to(Config['device'])


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


optimizer = optim.Adam(model.parameters(), lr=Config['lr'],
                       weight_decay=Config['weight_decay'])
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config['T_max'],
                                           eta_min=Config['min_lr'])


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(train_df) // Config['train_batch_size'])

    for step, x in bar:
        videos = [video.to(Config['device']) for video in x['video']]
        labels = x['label'].to(device, dtype=torch.long)

        batch_size = len(videos)
        outputs = model(videos)
        loss = criterion(outputs=outputs, labels=labels)
        loss = loss / Config['n_accumulate']

        loss.backward()

        if (step + 1) % Config['n_accumulate'] == 0:
            optimizer.step()

            # Zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        bar.set_postfix(
            Epoch=epoch, Train_Loss=epoch_loss,
            LR=optimizer.param_groups[0]['lr']
        )
    gc.collect()

    return epoch_loss


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    targets = []
    outs = []

    bar = tqdm(enumerate(dataloader), total=len(val_df) // Config["valid_batch_size"])
    for step, x in bar:
        videos = [video.to(Config['device']) for video in x["video"]]
        labels = x['label'].to(device, dtype=torch.long)

        batch_size = len(videos)

        outputs = model(videos)
        loss = criterion(outputs, labels)
        outs.append(outputs.detach())
        targets.extend(labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    outs = torch.vstack(outs)
    outs = outs.to('cpu')
    targets = torch.tensor(targets, device='cpu')
    epoch_acc = accuracy(outs, targets, topk=(1, 5))
    gc.collect()

    return epoch_loss, epoch_acc


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (
                y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(
                -1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0,
                                                                                        keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    #     wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_5_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler,
                                           dataloader=train_loader,
                                           device=Config['device'], epoch=epoch)

        val_epoch_loss, val_acc = valid_one_epoch(model, val_loader,
                                                  device=Config['device'],
                                                  epoch=epoch)

        val_1_acc, val_5_acc = val_acc[0].item(), val_acc[1].item()

        #         # Log the metrics
        #         wandb.log({"Train Loss": train_epoch_loss})
        #         wandb.log({"Valid Loss": val_epoch_loss})
        #         wandb.log({"Top 1 Accuracy": val_1_acc})
        #         wandb.log({"Top 5 Accuracy": val_5_acc})

        # deep copy the model
        if val_5_acc >= best_5_acc:
            print(f"{b_}Validation Acc Improved ({best_5_acc} ---> {val_5_acc})")
            best_5_acc = val_5_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "Acc{:.4f}_epoch{:.0f}.bin".format(best_5_acc, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Top5 Accuracy: {:.4f}".format(best_5_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


model = run_training(model, optimizer, scheduler,
                     device=Config['device'],
                     num_epochs=Config['epochs'])
