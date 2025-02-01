import torch 
import wandb
from scripts.train import train
from scripts.eval import eval 

class Trainer():
    def __init__(self, num_epochs, criterion, optimizer, scheduler, patience, save_every, model_name, start=0, best_val_acc=0, device="mps"):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.start = start
        self.device = device
        self.best_val_acc = best_val_acc
        self.patience = patience
        self.save_every = save_every
        self.model_name = model_name

    def fit(self, model, train_loader, val_loader, early_stopping=True, log=True, save_best=True, checkpoints=True):
        patience = 0
        best_epoch = 0

        for epoch in range(self.start, self.num_epochs):
            print("\nEpoch {}/{}".format(epoch+1, self.num_epochs))

            curr_lr                 = float(self.optimizer.param_groups[0]['lr'])
            train_loss, train_acc   = train(model, train_loader, self.optimizer, self.criterion, self.device)
            val_loss, val_acc       = eval(model, val_loader, self.criterion, self.device)

            print("\tTrain Acc {:.04f}%\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_acc*100, train_loss, curr_lr))
            print("\tVal Acc {:.04f}%\tVal Loss {:.04f}".format(val_acc*100, val_loss))

            if log:
                ## Log metrics at each epoch in your run
                # Optionally, you can log at each batch inside train/eval functions
                # (explore wandb documentation/wandb recitation)
                wandb.log({'train_acc': train_acc*100, 'train_loss': train_loss,
                        'val_acc': val_acc*100, 'valid_loss': val_loss, 'lr': curr_lr})

            self.scheduler.step(val_acc)

            if early_stopping:
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    patience = 0
                    best_epoch = epoch + 1

                    if save_best:
                        torch.save(model.state_dict(), f'checkpoints/{self.model_name}/best_model.pth') 
                else:
                    patience += 1

                if self.patience >= self.patience:
                    print(f"Best epoch was {best_epoch} with val_acc={self.best_val_acc*100:.2f}%.")
                    break 
            
            if checkpoints:
                if (epoch+1) % self.save_every == 0:
                    checkpoint_path = f"checkpoints/{self.model_name}/model_epoch_{epoch+1}.pth"
                    torch.save(model.state_dict(), checkpoint_path)

