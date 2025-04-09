from scripts.train_val import train_model, validate_model
import wandb
import os 
import torch

class Trainer():
    def __init__(self, num_epochs, criterion, optimizer, decoder, scheduler, LABELS, save_every, model_name, scaler, checkpoint_dir, start=0, device="cuda", best_valid_dist=float('inf')):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.decoder = decoder
        self.LABELS = LABELS
        self.criterion = criterion
        self.start = start
        self.device = device
        self.best_epoch = start
        self.scheduler = scheduler
        self.save_every = save_every
        self.model_name = model_name
        self.scaler = scaler
        self.checkpoint_dir = checkpoint_dir
        self.best_valid_dist = best_valid_dist

    def fit(self, model, train_loader, val_loader, log_epoch=True, log_batch=True, log_freq=10, save_best=True, checkpoints=True):

        model = model.to(self.device)

        for epoch in range(self.start, self.num_epochs):

            print("\nEpoch {}/{}".format(epoch+1, self.num_epochs))

            curr_lr                 = float(self.optimizer.param_groups[0]['lr'])
            train_loss              = train_model(model, train_loader, self.criterion, self.scheduler, self.optimizer, self.scaler, self.device, log_batch, log_freq)
            valid_loss, valid_dist  = validate_model(model, val_loader, self.criterion, self.decoder, self.LABELS, self.device)

            print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
            print("\tVal Dist {:.04f}\t Val Loss {:.04f}".format(valid_dist, valid_loss))

            
            # Logging
            if log_epoch:
                ## Log metrics at each epoch in the run
                wandb.log({
                    'train_loss': train_loss,
                    'valid_dist': valid_dist,
                    'valid_loss': valid_loss,
                    'lr': curr_lr
                })

            if checkpoints:
                os.makedirs(f'{self.checkpoint_dir}/{self.model_name}', exist_ok=True) 

            if checkpoints:
                    if (epoch+1) % self.save_every == 0:
                        checkpoint_path = f"{self.checkpoint_dir}/{self.model_name}/model_epoch_{epoch+1}.pth"
                        
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                            'loss': valid_loss, 
                            "dist": valid_dist
                        }

                        torch.save(checkpoint, checkpoint_path)
            
            if save_best and valid_dist < self.best_valid_dist:
                self.best_valid_dist = valid_dist
                self.best_epoch = epoch + 1
                os.makedirs(f'{self.checkpoint_dir}/{self.model_name}', exist_ok=True) 
                checkpoint_path = f"{self.checkpoint_dir}/{self.model_name}/best_model.pth"
                
                checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                            'valid_dist': valid_dist
                        }
                torch.save(checkpoint, checkpoint_path)




