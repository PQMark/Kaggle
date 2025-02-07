from torch.utils.data import DataLoader, TensorDataset
import torchaudio.transforms as tat
import os 
from tqdm import tqdm
import numpy as np 
import torch 
import torch.nn.functional as F

class DataModule:
    def __init__(self, root, batch_size, num_workers=0, pin_memory=True):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def initialize(self, mode):
        """
        load datasets:
        self.train_data
        self.val_data
        self.test_data
        """
        raise NotImplementedError
    
    def get_dataloader(self, train):
        raise NotImplementedError
    
    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def test_dataloader(self):
        """
        no labels
        """
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False, 
            pin_memory=self.pin_memory
        )

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """
        tensors = (X, y)
        """
        tensors = tuple(t[indices] for t in tensors)
        dataset = TensorDataset(*tensors)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train)


class AudioDataset:
    def __init__(self, root, phonemes, partition, config, load_transcript=True, test=False):
        self.config = config
        self.context    = config["context"]
        self.phonemes   = phonemes
        self.subset = config["subset"]
        self.load_transcript = load_transcript

        # Initialize the map from string to ID
        self.phonemes_map = {phoneme : id for id, phoneme in enumerate(phonemes)}

        # Initialize augmentation transforms
        if config["augmentations"] in ["FreqMask", "Both"]:
          self.freq_masking = tat.FrequencyMasking(config["freq_mask_param"])
        
        if config["augmentations"] in ["TimeMask", "Both"]:
          self.time_masking = tat.TimeMasking(config["time_mask_param"])

        # Initialize paths
        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.transcript_dir = os.path.join(root, partition, "transcript")

        # List MFCC files in sorted order
        if test:
            mfcc_names = sorted(os.listdir(self.mfcc_dir))
        else:
           mfcc_names = sorted(os.listdir(self.mfcc_dir), key=lambda x: [int(f) for f in x.replace(".npy", "").split('-')])

        # Subset
        subset_size = int(self.subset * len(mfcc_names))
        mfcc_names = mfcc_names[:subset_size]


        # List transcripts files in sorted order
        if self.load_transcript:
            transcript_names = sorted(os.listdir(self.transcript_dir), key=lambda x: [int(f) for f in x.replace(".npy", "").split('-')])      # sorted(os.listdir(self.transcript_dir))
            transcript_names = transcript_names[:subset_size]
            assert len(mfcc_names) == len(transcript_names)
        
        # store MFCC and transcripts 
        self.mfccs, self.transcripts = [], []
        for i in tqdm(range(len(mfcc_names))):
            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))
            mfccs_normalized = mfcc - np.mean(mfcc, axis=0, keepdims=True) # Cepstral Normalization at Time Dimension
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)
            self.mfccs.append(mfccs_normalized)

            # Load the corresponding transcript & Remove [SOS] and [EOS] from the transcript
            if self.load_transcript:
                transcript = np.load(os.path.join(self.transcript_dir, transcript_names[i]))
                transcript  = transcript[1:-1]
            
                # map to ID 
                transcript_indices = np.array([self.phonemes_map[p] for p in transcript]) 
                transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)
                self.transcripts.append(transcript_indices)
        

        # Concatenate all mfccs
        self.mfccs= torch.cat(self.mfccs, 0)    # shape: T x 28 (Where T = T1 + T2 + ...)

        # Concatenate all transcripts
        if self.load_transcript:
          self.transcripts = torch.cat(self.transcripts, 0)   # shape: (T,)

        self.length = len(self.mfccs)

        # introduce context by padding zeros on top and bottom of self.mfcc
        self.mfccs = F.pad(self.mfccs, (0, 0, config["context"], config["context"]), mode="constant")
    
    def collate_fn(self, batch):
      x, y = zip(*batch)
      x = torch.stack(x, dim=0)   # shape: (batch_size, time, freq)

      # Apply augmentations with 70% probability
      if np.random.rand() < 0.70:
        x = x.transpose(1, 2)  # Shape: (batch_size, freq, time)
        x = getattr(self, "freq_masking", lambda x: x)(x)
        x = getattr(self, "time_masking", lambda x: x)(x)
        x = x.transpose(1, 2)  # Shape back to: (batch_size, time, freq)

      return x, torch.tensor(y)

    def __len__(self):
        return self.length
    
    def __getitem__(self, ind):
        start = ind 
        end = ind + self.context * 2 + 1
        frames = self.mfccs[start:end]

        if self.load_transcript:
          phonemes = self.transcripts[ind]
          return frames, phonemes
        else:
           return frames

class AudioDatasetModule(DataModule):
    def __init__(self, root, phonemes, train_partition, val_partition, test_partition, batch_size, config, num_workers=0, pin_memory=True):
      super().__init__(root, batch_size, num_workers, pin_memory)
      self.phonemes = phonemes
      self.train_partition = train_partition
      self.val_partition = val_partition
      self.test_partition = test_partition
      self.config = config
    
    def initialize(self, mode):
        if mode == "fit":
          self.train_data = AudioDataset(
             root=self.root,
             phonemes=self.phonemes,
             partition=self.train_partition,
             config=self.config,
             load_transcript=True)
          
          self.val_data = AudioDataset(
             root=self.root, 
             phonemes=self.phonemes,
             partition=self.val_partition,
             config=self.config,
             load_transcript=True)
          
        elif mode == "test":
            self.test_data = AudioDataset(
                root=self.root, 
                phonemes=self.phonemes,
                partition=self.test_partition,
                config=self.config,
                load_transcript=False, 
                test=True
            )

    def get_dataloader(self, train):
       return DataLoader(
          self.train_data if train else self.val_data, 
          batch_size=self.batch_size, 
          num_workers=self.num_workers, 
          shuffle=train, 
          pin_memory=self.pin_memory, 
          collate_fn=self.train_data.collate_fn if train else None
       )
