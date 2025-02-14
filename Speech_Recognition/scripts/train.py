from tqdm import tqdm
import torch 
import wandb

def train(model, dataloader, scheduler, optimizer, criterion, device="mps", log_batch=True, log_freq=10, scaler=None, max_grad_norm=1.0):
    '''
    log_batch: whether log for batches
    '''

    model.train()
    tloss, tacc = 0, 0 

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    for i, (frames, phonemes) in enumerate(dataloader):
        
        # if i == len(dataloader) // 3:
        #     torch.save(model.state_dict(), f'checkpoints/model_d/intermediate_batch{i+1}_model.pth')

        # Initialize Gradients
        optimizer.zero_grad()

        frames = frames.to(device)
        phonemes = phonemes.to(device)

        # converts computations to a lower precision to speed up training and reduce memory usage
        
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits = model(frames)
            loss = criterion(logits, phonemes)

        if scaler is not None:
            # Backward Propagation
            scaler.scale(loss).backward()
            
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()

        batch_acc = torch.sum(torch.argmax(logits, dim= 1) == phonemes).item()/logits.shape[0]
        tloss += loss.item()
        tacc += batch_acc

        batch_bar.set_postfix(loss="{:.04f}".format(float(tloss / (i + 1))),
                            acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        batch_bar.update()

        if log_batch and i % log_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_dict = {
                "batch_loss": loss.item(),
                "batch_acc": batch_acc,
                "lr": current_lr,
                "batch": i
            }
            wandb.log(log_dict)

        # Release memory
        del frames, phonemes, logits

        if device.lower().startswith("cuda"):
            torch.cuda.empty_cache()
        elif device.lower().startswith("mps"):
            torch.mps.empty_cache()

    batch_bar.close()
    tloss   /= len(dataloader)
    tacc    /= len(dataloader)

    return tloss, tacc
