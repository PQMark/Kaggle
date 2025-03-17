from scripts.metrics import AverageMeter, accuracy
from tqdm import tqdm
import torch
import wandb

def train_epoch(model, dataloader, optimizer, criterion, lr_scheduler, scaler, device, log_batch, log_freq):

    model.train()

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad() # Zero gradients

        # send to cuda
        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        # forward   
        if scaler is not None:
            with torch.cuda.amp.autocast():  # This implements mixed precision. Thats it!
                outputs = model(images)

                # Use the type of output depending on the loss function you want to use
                loss = criterion(outputs['out'], labels)

            scaler.scale(loss).backward() # This is a replacement for loss.backward()
            scaler.step(optimizer) # This is a replacement for optimizer.step()
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs['out'], labels)
            loss.backward()
            optimizer.step()

        # metrics
        loss_m.update(loss.item())
        if 'feats' in outputs:
            acc = accuracy(outputs['out'], labels)[0].item()
        else:
            acc = 0.0
        acc_m.update(acc)


        batch_bar.set_postfix(
            # acc         = "{:.04f}%".format(100*accuracy),
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss        = "{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr          = "{:.04f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update() 

        # You may want to call some schedulers inside the train function. What are these?
        # if lr_scheduler is not None:
        #     lr_scheduler.step()

        if log_batch and i % log_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_dict = {
                "batch_loss": loss.item(),
                "batch_acc": acc,
                "lr": current_lr,
                "batch": i
            }
            wandb.log(log_dict)

        
        
    batch_bar.close()

    return acc_m.avg, loss_m.avg