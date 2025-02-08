from tqdm import tqdm
import torch 

def eval(model, dataloader, criterion, device="mps"):

    model.eval()
    vloss, vacc = 0, 0 
    batch_bar   = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    for i, (frames, phonemes) in enumerate(dataloader):

        frames      = frames.to(device)
        phonemes    = phonemes.to(device)

        with torch.inference_mode():
            outputs = model(frames)

            ### Forward Propagation
            if isinstance(outputs, (tuple, list)):
                main_logits = outputs[0]
            else:
                main_logits = outputs

            ### Loss Calculation
            loss    = criterion(main_logits, phonemes)
        
        vloss   += loss.item()
        vacc    += torch.sum(torch.argmax(main_logits, dim= 1) == phonemes).item()/main_logits.shape[0]

        batch_bar.set_postfix(loss="{:.04f}".format(float(vloss / (i + 1))),
                              acc="{:.04f}%".format(float(vacc*100 / (i + 1))))
        batch_bar.update()

        ### Release memory
        del frames, phonemes, main_logits
        if device.lower().startswith("cuda"):
            torch.cuda.empty_cache()
        elif device.lower().startswith("mps"):
            torch.mps.empty_cache()

    batch_bar.close()
    vloss   /= len(dataloader)
    vacc    /= len(dataloader)

    return vloss, vacc