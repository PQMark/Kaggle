import torch
import random
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
# from torchsummaryX import summary
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import wandb
import torchaudio.transforms as tat
from torchaudio.models.decoder import cuda_ctc_decoder

from sklearn.metrics import accuracy_score
import gc

import sys 
sys.path.append('/home/pengq/speech_recognition')

from scripts.data_prep import AudioDataset
from models.ASR import ASRModel, initialize_weights
from models.AASR import AASRModel
from scripts.train_val import calculate_levenshtein
from scripts.trainer import Trainer


import glob

import zipfile
from tqdm.auto import tqdm
import os
import datetime
import argparse

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='run.1', help='Name for this run')
args = parser.parse_args()


model_name = "AASR"                                     # CHECK THIS
name = f"{model_name}_{args.run_name}"
num_worker = 8
start=0
load_cpk = ""           # /data/user_data/pengq/speech/AASR_run4/best_model.pth
wandb_id = "gnvb4rii"
load_optimizer = True
load_scheduler = True
resume_start = True
best_dist = float('inf')        # default

config = {
    "subset": 1.0, 
    "epochs": 100,
    "batch_size": 128, 
    "augmentations": "TimeMask", 
    "time_mask_param": 10, 
    "freq_mask_param": 10,
    "mfcc_features": 28,

    "embed_size": 400,  # 384                           # CHECK THIS 
    "cnn_arch": [[128, 3, 1, 1], [256, 3, 1, 1]],
    "linear_arch": [[2048, 0.2], [1024, 0.2]],
    "num_pBLSTM": 2,
    "encoder_dropout": 0.2,
    "lstm_dropout": 0.2,
    "decoder_dropout": 0.2,
    "input_size": 28,

    "lr": 0.001,
    "weight_decay": 1e-5,
    "pct_start": 0.25,
    "final_div_factor": 100,
    "train_beam_width": 10,
    "test_beam_width": 10,

    "save_every": 2, 
}


CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" :
     "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict[:-2]     #To be used for mapping original transcripts to integer indices
LABELS = ARPAbet[:-2]       #To be used for mapping predictions to strings

OUT_SIZE = len(PHONEMES)    # Number of output classes
print("Number of Phonemes:", OUT_SIZE)

# Indexes of BLANK and SIL phonemes
BLANK_IDX=CMUdict.index('')
SIL_IDX=CMUdict.index('[SIL]')

print("Index of Blank:", BLANK_IDX)
print("Index of [SIL]:", SIL_IDX)

root = "/data/user_data/pengq/11785-S25-hw3p2"

test_mfcc = f"{root}/train-clean-100/mfcc/103-1240-0000.npy"
test_transcript = f"{root}/train-clean-100/transcript/103-1240-0000.npy"

mfcc = np.load(test_mfcc)
transcript = np.load(test_transcript)[1:-1] #Removed [SOS] and [EOS]

print("MFCC Shape:", mfcc.shape)
print("\nMFCC:\n", mfcc)
print("\nTranscript shape:", transcript.shape)

print("\nOriginal Transcript:\n", transcript)

# map the loaded transcript (from phonemes representation) to corresponding labels representation
mapped_transcript = [CMUdict_ARPAbet[k] for k in transcript]
print("\nTranscript mapped from PHONEMES representation to LABELS representation:\n", mapped_transcript)

# Mapping list of PHONEMES to list of Integer indexes
map = {k: i for i, k in enumerate(PHONEMES)}
print("\nMapping list of PHONEMES to list of Integer indexes:\n", map)

import gc
gc.collect()

# Create objects for the dataset classes
train_data = AudioDataset(root=root, PHONEMES=PHONEMES, partition="train-clean-100", config=config)
val_data = AudioDataset(root=root, PHONEMES=PHONEMES, partition="dev-clean", config=config)
test_data = AudioDataset(root=root, PHONEMES=PHONEMES, partition="test-clean", config=config, load_transcript=False, test=True)

train_loader = torch.utils.data.DataLoader(
    dataset     = train_data,
    num_workers = num_worker,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = True,
    collate_fn = train_data.collate_fn
) 

val_loader = torch.utils.data.DataLoader(
    dataset     = val_data,
    num_workers = num_worker,
    batch_size  = config['batch_size'],
    pin_memory  = True,
    shuffle     = False,
    collate_fn=val_data.collate_fn
) 

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    num_workers=num_worker,
    batch_size=config['batch_size'],
    pin_memory=True,
    collate_fn=test_data.collate_fn,
    shuffle=False
)

print("Batch size: ", config['batch_size'])
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))
print("Test dataset samples = {}, batches = {}".format(test_data.__len__(), len(test_loader)))

# sanity check
for data in train_loader:
    x, y, lx, ly = data
    print(x.shape, y.shape, lx.shape, ly.shape)
    break

# model = ASRModel(
#     cnn_arch = config["cnn_arch"], 
#     input_size = config["input_size"], 
#     encoder_hidden_size = config["embed_size"], 
#     num_pBLSTM = config["num_pBLSTM"], 
#     linear_arch = config["linear_arch"], 
#     embed_size = config["embed_size"], 
#     encoder_dropout = config["encoder_dropout"], 
#     decoder_dropout = config["decoder_dropout"]
# ).to(device)

# model.apply_init(x.to(device), lx.to(device), initialize_weights)


model = AASRModel(
    input_size      = config["input_size"],
    embed_size      = config["embed_size"], 
    linear_arch     = config["linear_arch"], 
    decoder_dropout = config["decoder_dropout"]
).to(device)
model.apply_init(x.to(device), lx.to(device), initialize_weights)

summary(model, input_data=[x.to(device), lx.to(device)])

# model.apply_init(x.to(device), lx.to(device), initialize_weights)
criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config["train_beam_width"])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"], 
        total_steps = config["epochs"] * len(train_loader), 
        pct_start = config["pct_start"], 
        anneal_strategy="cos", 
        final_div_factor=config["final_div_factor"]
    )

scaler = torch.cuda.amp.GradScaler()

torch.cuda.empty_cache()
gc.collect()

if load_cpk != "":
    checkpoint = torch.load(load_cpk)
    
    print(f"\nLoad checkpoint from {load_cpk}")
    print("Load model checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])

    if load_optimizer:
        print("Load optimier checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if load_scheduler:
        print("Load scheduler checkpoint")
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if resume_start:
        start = checkpoint["epoch"]
        print(f"Start from {start}")
    
    best_dist = checkpoint['valid_dist']
    
    print(f"Curr Val Dist: {checkpoint['valid_dist']}")


wandb.login(key="232fdceeae8cfc797a73cea836fb0c4a6199a7ae")

if load_cpk != "":
    run = wandb.init(
        name    = name, ### set run names
        reinit  = True, ### Allows reinitalizing runs when re-running this cell
        id      = wandb_id, ### Insert specific run id here if resuming a previous run
        resume  = "must", ### need this to resume previous runs, but comment out reinit = True when using this
        project = "HW3P2", ### Project name
        group=f"{model_name}", 
        config=config, 
        entity="11785-DL"
    )
else:
    run = wandb.init(
        name    = name, ### set run names
        reinit  = True, ### Allows reinitalizing runs when re-running this cell
        #id     = "", ### Insert specific run id here if resuming a previous run
        #resume = "must", ### need this to resume previous runs, but comment out reinit = True when using this
        project = "HW3P2", ### Project name
        group=f"{model_name}", 
        config=config, 
        entity="11785-DL"
    )

test_Trainer = Trainer(config["epochs"], criterion, optimizer, decoder, scheduler, LABELS,
                       config["save_every"], name, device=device, scaler=scaler, checkpoint_dir="/data/user_data/pengq/speech", best_valid_dist=best_dist, start=start)
test_Trainer.fit(model, train_loader, val_loader, log_epoch=True, log_batch=False, save_best=True,
                 checkpoints=False)

