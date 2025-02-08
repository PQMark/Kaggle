from tqdm import tqdm
import torch 
import wandb

def train(model, dataloader, scheduler, optimizer, criterion, device="mps", log_batch=True, log_freq=10, scaler=None):
    '''
    log_batch: whether log for batches
    '''

    lambda_aux = 0.4

    model.train()
    tloss, tacc = 0, 0 
    aloss = 0

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
            outputs = model(frames)
        
        if isinstance(outputs, (tuple, list)):
            main_logits, aux_logits = outputs
            main_loss = criterion(main_logits, phonemes)
            aux_loss = criterion(aux_logits, phonemes)
            loss = main_loss + lambda_aux * aux_loss
        else:
            main_logits = outputs
            loss = criterion(outputs, phonemes)
            aux_loss = None

        # gradient clipping if exploding gradients is an issue
        # ......

        if scaler is not None:
            # Backward Propagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()

        batch_acc = torch.sum(torch.argmax(main_logits, dim= 1) == phonemes).item()/main_logits.shape[0]
        tloss += loss.item()
        tacc += batch_acc
        if aux_loss is not None:
            aloss += aux_loss.item()

        if aux_loss is not None:
            batch_bar.set_postfix(aux_loss="{:.04f}".format(float(aloss / (i +1))),
                                loss="{:.04f}".format(float(tloss / (i + 1))),
                                acc="{:.04f}%".format(float(tacc*100 / (i + 1))))
        else:
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
            if aux_loss is not None:
                log_dict["batch_aux_loss"] = aux_loss.item()
            wandb.log(log_dict)

        # Release memory
        del frames, phonemes, main_logits
        if aux_loss is not None:
            del aux_loss

        if device.lower().startswith("cuda"):
            torch.cuda.empty_cache()
        elif device.lower().startswith("mps"):
            torch.mps.empty_cache()

    batch_bar.close()
    tloss   /= len(dataloader)
    tacc    /= len(dataloader)

    return tloss, tacc
