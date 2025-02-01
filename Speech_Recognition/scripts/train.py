from tqdm import tqdm
import torch 

def train(model, dataloader, optimizer, criterion, device="mps"):

    model.train()
    tloss, tacc = 0, 0 
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    for i, (frames, phonemes) in enumerate(dataloader):

        # Initialize Gradients
        optimizer.zero_grad()

        frames = frames.to(device)
        phonemes = phonemes.to(device)

        # converts computations to a lower precision to speed up training and reduce memory usage
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(frames)
            loss = criterion(logits, phonemes)
        
        # Backward Propagation
        loss.backward()

        # gradient clipping if exploding gradients is an issue
        # ......

        optimizer.step()

        tloss += loss.item()
        tacc += torch.sum(torch.argmax(logits, dim= 1) == phonemes).item()/logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                              acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        batch_bar.update()

        # Release memory
        del frames, phonemes, logits
        torch.mps.empty_cache()

    batch_bar.close()
    tloss   /= len(dataloader)
    tacc    /= len(dataloader)

    return tloss, tacc
