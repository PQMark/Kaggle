import torch
import os
from tqdm import tqdm
import numpy as np
import torchaudio.transforms as tat
from torch.nn.utils.rnn import pad_sequence

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, root, PHONEMES, partition, config, load_transcript=True, test=False):
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''

        self.PHONEMES = PHONEMES
        self.subset = config['subset']
        self.load_transcript = load_transcript
        self.test = test
        
        # Initialize the map from string to ID
        self.phonemes_map = {phoneme : id for id, phoneme in enumerate(self.PHONEMES)}
        
        if config["augmentations"] in ["FreqMask", "Both"]:
          self.freq_masking = tat.FrequencyMasking(config["freq_mask_param"])
        
        if config["augmentations"] in ["TimeMask", "Both"]:
          self.time_masking = tat.TimeMasking(config["time_mask_param"])

        # Define the directories containing MFCC and transcript files
        self.mfcc_dir = os.path.join(root, partition, "mfcc")
        self.transcript_dir = os.path.join(root, partition, "transcript")
    
        # List all files in the directories. Remember to sort the files
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))
        
        # Subset
        subset_size = int(self.subset * len(self.mfcc_files))
        self.mfcc_files = self.mfcc_files[:subset_size]

        if self.load_transcript:
            self.transcript_files = sorted(os.listdir(self.transcript_dir))
            self.transcript_files = self.transcript_files[:subset_size]
            assert(len(self.mfcc_files) == len(self.transcript_files))

        self.length = len(self.mfcc_files)

        # store MFCC and transcripts 
        self.mfccs, self.transcripts = [], []
        for i in tqdm(range(len(self.mfcc_files))):
            mfcc = np.load(os.path.join(self.mfcc_dir, self.mfcc_files[i]))
            mfcc_mean = np.mean(mfcc, axis=0, keepdims=True)
            mfcc_std = np.std(mfcc, axis=0, keepdims=True) + 1e-8
            mfccs_normalized = (mfcc - mfcc_mean) / mfcc_std

            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)
            self.mfccs.append(mfccs_normalized)

            # Load the corresponding transcript & Remove [SOS] and [EOS] from the transcript
            if self.load_transcript:
                transcript = np.load(os.path.join(self.transcript_dir, self.transcript_files[i]))
                transcript  = transcript[1:-1]

                # map to ID 
                transcript_indices = np.array([self.phonemes_map[p] for p in transcript]) 
                transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)
                self.transcripts.append(transcript_indices)


    def __len__(self):
        return self.length


    def __getitem__(self, ind):
        '''
        RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS
        '''
        mfcc = self.mfccs[ind]
        
        if self.load_transcript:
            transcript = self.transcripts[ind]
            return mfcc, transcript
        
        return mfcc


    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish.
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features,
            and lengths of labels.
        '''

        # Extract batch of input MFCCs and batch of output transcripts separately
        if self.load_transcript:
            batch_mfcc = [sample[0] for sample in batch]
            batch_transcript = [sample[1] for sample in batch]
        else:
            batch_mfcc = batch

        # Store original lengths of the MFCCS and transcripts in the batches
        lengths_mfcc = [mfcc.shape[0] for mfcc in batch_mfcc]
        if self.load_transcript:
            lengths_transcript = [transcript.shape[0] for transcript in batch_transcript]

        # Pad the MFCC sequences and transcripts
        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        # Note: (resulting shape of padded MFCCs: [batch, time, freq])
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True)
        if self.load_transcript:
            batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True)

        # Apply augmentations with 70% probability
        if np.random.rand() < 0.70 and not self.test:
            if hasattr(self, "freq_masking"):
                batch_mfcc_pad = batch_mfcc_pad.transpose(1, 2)
                batch_mfcc_pad = self.freq_masking(batch_mfcc_pad)
                batch_mfcc_pad = batch_mfcc_pad.transpose(1, 2)

            if hasattr(self, "time_masking"):
                batch_mfcc_pad = batch_mfcc_pad.transpose(1, 2)
                batch_mfcc_pad = self.time_masking(batch_mfcc_pad)
                batch_mfcc_pad = batch_mfcc_pad.transpose(1, 2)

        # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
        if self.load_transcript:
            return batch_mfcc_pad, batch_transcript_pad, torch.tensor(lengths_mfcc), torch.tensor(lengths_transcript)
        else:
            return batch_mfcc_pad, torch.tensor(lengths_mfcc)
