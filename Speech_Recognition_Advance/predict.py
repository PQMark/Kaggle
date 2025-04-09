import sys 
sys.path.append('/home/pengq/speech_recognition')

from models.ASR import ASRModel, initialize_weights
from models.AASR import AASRModel
from scripts.train_val import decode_prediction

from torch.utils.data import Dataset, DataLoader
from scripts.data_prep import AudioDataset
from torchaudio.models.decoder import cuda_ctc_decoder
import torch
from tqdm import tqdm
import pandas as pd

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

LABELS = ARPAbet[:-2]
PHONEMES = CMUdict[:-2]

root = "/data/user_data/pengq/11785-S25-hw3p2"
num_worker = 6

config = {
    "test_beam_width": 10, 
    "input_size": 28, 
    "embed_size": 384, 
    "augmentations": "", 
    "subset": 1.0, 
    "batch_size": 128
}

test_data = AudioDataset(root=root, PHONEMES=PHONEMES, partition="test-clean", config=config, load_transcript=False, test=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    num_workers=num_worker,
    batch_size=config['batch_size'],
    pin_memory=True,
    collate_fn=test_data.collate_fn,
    shuffle=False
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

load_cpk = "/data/user_data/pengq/speech/AASR_run4.2/best_model.pth"
model = AASRModel(
    input_size  = config["input_size"],
    embed_size  = config["embed_size"]
).to(device)

checkpoint = torch.load(load_cpk)

print(f"Curr Val Dist: {checkpoint['valid_dist']}")
model.load_state_dict(checkpoint["model_state_dict"])

test_decoder = cuda_ctc_decoder(tokens=LABELS, nbest=1, beam_size=config['test_beam_width']) 
results = []
model.eval()
print("Testing")

for data in tqdm(test_loader):
    x, lx = data
    x = x.to(device)

    lx = lx.to(device) 

    with torch.no_grad():
        h, lh = model(x, lx)

        h = h.to(device).contiguous()
        lh = lh.to(device).contiguous()

        print(f"h device: {h.device}, lh device: {lh.device}")

        prediction_string = decode_prediction(h, lh, test_decoder, LABELS)
        results.extend(prediction_string)

    del x, lx, h, lh
    torch.cuda.empty_cache()


if results:
    df = pd.DataFrame({
        'index': range(len(results)), 'label': results
    })

data_dir = "submission.csv"
df.to_csv(data_dir, index = False)


