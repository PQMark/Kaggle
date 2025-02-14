# Speech Recognition 

### Data Preprocessing
MFCC data is Cepstral normalized along the time dimension, and is zero padded to the both ends to facilitate context division. Transcripts are cleaned by removing the start ([SOS]) and end ([EOS]) tokens. Phonemes in the transcripts are mapped to discrete numbers. During training, with a 70% probability, frequency masking or time masking are applied to the MFCCs to augment the data for robustness. 

### Model Artitecture
<img src="figs/architecture.png" alt="model archeitecture" width="600"/>

SiLu activation is used. Dropout layer is added for A[1] - A[5] layers. 

### Hyperparameters
Layer outputs: [4096, 2048, 1024, 1024, 750, 512, 42]    
Dropout rates: [0.2, 0.15, 0.15, ... ...]

number of epoches

