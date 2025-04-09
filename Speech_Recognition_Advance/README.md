# Speech Recognition 

This project implements an automatic speech recognition (ASR) system that maps audio utterances to phoneme sequences using deep learning. It explores RNN-based architectures (LSTM/GRU), CNNs, and pyramidal BiLSTMs for feature extraction and temporal modeling.  

Mel-Frequency Cepstral Coefficients (MFCCs) serve as the input features, and the model is trained with Connectionist Temporal Classification (CTC) loss to handle the alignment between variable-length input and output sequences. Inference uses beam search decoding, while performance is evaluated using the Levenshtein distance on phoneme sequences.

---

## Data Preprocessing
- **MFCC Features:**    
The MFCC data is Cepstral normalized along the time axis, and zero-padded at both ends to help with context division.

- **Transcripts:**  
Transcripts are cleaned by removing the start (`[SOS]`) and end (`[EOS]`) tokens. Each phoneme in the transcript is then mapped to a discrete number.

- **Data Augmentation:**  
 During training, with a 70% probability, frequency masking or time masking are applied to the MFCCs to augment the data for robustness. 


## Model Architecture
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
AASRModel                                [28, 408, 41]             --
├─Encoder: 1-1                           [28, 408, 800]            --
│    └─Conv1d: 2-1                       [28, 128, 1633]           10,880  
│    └─BatchNorm1d: 2-2                  [28, 128, 1633]           256
│    └─GELU: 2-3                         [28, 128, 1633]           --
│    └─Dropout: 2-4                      [28, 128, 1633]           --
│    └─TDNNResBlock: 2-5                 [28, 128, 1633]           --
│    │    └─TDNNLayer: 3-1               [28, 128, 1633]           49,408
│    │    └─TDNNLayer: 3-2               [28, 128, 1633]           49,408
│    │    └─Identity: 3-3                [28, 128, 1633]           --
│    │    └─Dropout: 3-4                 [28, 128, 1633]           --
│    └─TDNNResBlock: 2-6                 [28, 256, 1633]           --
│    │    └─TDNNLayer: 3-5               [28, 256, 1633]           98,816
│    │    └─TDNNLayer: 3-6               [28, 256, 1633]           197,120
│    │    └─Sequential: 3-7              [28, 256, 1633]           33,280
│    │    └─Dropout: 3-8                 [28, 256, 1633]           --
│    └─TDNNResBlock: 2-7                 [28, 512, 1633]           --
│    │    └─TDNNLayer: 3-9               [28, 512, 1633]           394,240
│    │    └─TDNNLayer: 3-10              [28, 512, 1633]           787,456
│    │    └─Sequential: 3-11             [28, 512, 1633]           132,096
│    │    └─Dropout: 3-12                [28, 512, 1633]           --
│    └─Conv1d: 2-8                       [28, 512, 1633]           65,536
│    └─Dropout: 2-9                      [28, 512, 1633]           --
│    └─ResidualPBLSTM: 2-10              [18960, 800]              --
│    │    └─LSTM: 3-13                   [18960, 800]              4,563,200
│    │    └─Linear: 3-14                 [28, 816, 800]            410,400
│    │    └─LayerNorm: 3-15              [28, 816, 800]            1,600
│    │    └─Dropout: 3-16                [28, 816, 800]            --
│    └─LockedDropout: 2-11               [18960, 800]              --
│    └─ResidualPBLSTM: 2-12              [9473, 800]               --
│    │    └─LSTM: 3-17                   [9473, 800]               6,406,400
│    │    └─LayerNorm: 3-18              [28, 408, 800]            1,600
│    │    └─Dropout: 3-19                [28, 408, 800]            --
│    └─LockedDropout: 2-13               [9473, 800]               --
│    └─LayerNorm: 2-14                   [28, 408, 800]            1,600
│    └─Dropout: 2-15                     [28, 408, 800]            --
├─Decoder: 1-2                           [28, 408, 41]             --
│    └─Sequential: 2-16                  [28, 408, 41]             --
│    │    └─Permute: 3-20                [28, 800, 408]            --
│    │    └─BatchNorm1d: 3-21            [28, 800, 408]            1,600
│    │    └─Permute: 3-22                [28, 408, 800]            --
│    │    └─Sequential: 3-23             [28, 408, 1024]           3,744,768
│    │    └─Linear: 3-24                 [28, 408, 41]             42,025
│    │    └─Dropout: 3-25                [28, 408, 41]             --
│    └─LogSoftmax: 2-17                  [28, 408, 41]             --
==========================================================================================
Total params: 10,270,633
Trainable params: 10,270,633
Non-trainable params: 0
Total mult-adds (Units.TERABYTES): 160.58
==========================================================================================
```

### Other Attempts 
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ASRModel                                 [128, 413, 41]            --
├─Encoder: 1-1                           [128, 413, 512]           --
│    └─Sequential: 2-1                   [128, 512, 1652]          --
│    │    └─Conv1d: 3-1                  [128, 128, 1652]          10,880
│    │    └─BatchNorm1d: 3-2             [128, 128, 1652]          256
│    │    └─GELU: 3-3                    [128, 128, 1652]          --
│    │    └─Conv1d: 3-4                  [128, 256, 1652]          98,560
│    │    └─BatchNorm1d: 3-5             [128, 256, 1652]          512
│    │    └─GELU: 3-6                    [128, 256, 1652]          --
│    │    └─Conv1d: 3-7                  [128, 512, 1652]          393,728
│    │    └─BatchNorm1d: 3-8             [128, 512, 1652]          1,024
│    │    └─GELU: 3-9                    [128, 512, 1652]          --
│    └─LSTMWrapper: 2-2                  [156617, 512]             --
│    │    └─LSTM: 3-10                   [156617, 512]             3,153,920
│    └─Sequential: 2-3                   [39105, 512]              --
│    │    └─pBLSTM: 3-11                 [78278, 512]              2,625,536
│    │    └─pBLSTM: 3-12                 [39105, 512]              2,625,536
│    └─LayerNorm: 2-4                    [128, 413, 512]           1,024
│    └─Dropout: 2-5                      [128, 413, 512]           --
├─Decoder: 1-2                           [128, 413, 41]            --
│    └─Sequential: 2-6                   [128, 413, 41]            --
│    │    └─Permute: 3-13                [128, 512, 413]           --
│    │    └─BatchNorm1d: 3-14            [128, 512, 413]           1,024
│    │    └─Permute: 3-15                [128, 413, 512]           --
│    │    └─Sequential: 3-16             [128, 413, 512]           5,256,704
│    │    └─Linear: 3-17                 [128, 413, 41]            21,033
│    │    └─Dropout: 3-18                [128, 413, 41]            --
│    └─LogSoftmax: 2-7                   [128, 413, 41]            --
==========================================================================================
Total params: 14,189,737
Trainable params: 14,189,737
Non-trainable params: 0
Total mult-adds (Units.TERABYTES): 410.81
==========================================================================================
```

---

## Hyperparameters  
- **Number of Epochs:** `100`  
- **Optimizer:** AdamW (weight_decay = `1e-5`)  
- **Learning Rate:** `1e-3`  
- **LR Scheduler:** OneCycleLR  
  Parameters: `max_lr=0.001`, `pct_start=0.25`, `anneal_strategy="cos"`  
- **Frequency Mask Parameter:** `10`
- **Batch Size:** `128`
- **Train Beam Width** `5`

---

## Experiment Records

- **Link to Wandb:**  
https://wandb.ai/11785-DL/HW3P2?nw=nwuserpeng_qiu

---

## Run the Code  
- The main script is in `main.py`. Use the `predict.py` to generate the predictions in csv format. 
- Use the `submit.sh` to submit jobs either for training or prediction. 