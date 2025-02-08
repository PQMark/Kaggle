import torch 
import wandb
from scripts.train import train
from scripts.eval import eval 
import os 

class Trainer():
    def __init__(self, num_epochs, criterion, optimizer, scheduler, patience, save_every, model_name, start=0, best_val_acc=0, device="mps", log_batch=True, log_freq=10, scaler=None):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.start = start
        self.device = device
        self.best_val_acc = best_val_acc
        self.best_epoch = start
        self.patience = patience
        self.scheduler = scheduler
        self.save_every = save_every
        self.model_name = model_name
        self.log_batch = log_batch
        self.log_freq = log_freq
        self.scaler = scaler

    def fit(self, model, train_loader, val_loader, early_stopping=True, log=True, save_best=True, checkpoints=True):

        model = model.to(self.device)
        current_patience = 0

        for epoch in range(self.start, self.num_epochs):

            print("\nEpoch {}/{}".format(epoch+1, self.num_epochs))

            curr_lr                 = float(self.optimizer.param_groups[0]['lr'])
            train_loss, train_acc   = train(model, train_loader, self.scheduler, self.optimizer, self.criterion, self.device, self.log_batch, self.log_freq)
            val_loss, val_acc       = eval(model, val_loader, self.criterion, self.device)

            print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc*100, train_loss, curr_lr))
            print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc*100, val_loss))

            if log:
                ## Log metrics at each epoch in your run
                # Optionally, you can log at each batch inside train/eval functions
                # (explore wandb documentation/wandb recitation)
                wandb.log({'train_acc': train_acc*100, 'train_loss': train_loss,
                        'val_acc': val_acc*100, 'valid_loss': val_loss, 'lr': curr_lr})
            
            if early_stopping or checkpoints:
                os.makedirs(f'checkpoints/{self.model_name}', exist_ok=True) 

            if checkpoints:
                    if (epoch+1) % self.save_every == 0:
                        checkpoint_path = f"checkpoints/{self.model_name}/model_epoch_{epoch+1}.pth"
                        
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                            'best_val_acc': self.best_val_acc
                        }

                        torch.save(checkpoint, checkpoint_path)

            if early_stopping:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    current_patience = 0
                    self.best_epoch = epoch + 1

                    if save_best:
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                            'best_val_acc': self.best_val_acc
                        }
                        torch.save(checkpoint, f'checkpoints/{self.model_name}/best_model.pth') 
                else:
                    current_patience += 1

                if current_patience >= self.patience:
                    print(f"Best epoch was {self.best_epoch} with val_acc={self.best_val_acc*100:.2f}%.")
                    break 

