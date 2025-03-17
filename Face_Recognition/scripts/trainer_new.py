from scripts.train import train_epoch
from scripts.val import valid_epoch_cls
from scripts.verification import valid_epoch_ver
import wandb
import os 
import torch
from perforatedai import pb_globals as PBG

class Trainer():
    def __init__(self, num_epochs, criterion, optimizer, scheduler, patience, save_every, model_name, start=0, best_val_acc=0, device="mps", scaler=None):
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
        self.scaler = scaler

    def fit(self, model, train_loader, val_loader, ver_loader, early_stopping=True, log_epoch=True, log_batch=True, log_freq=10, save_best=True, checkpoints=True):

        model = model.to(self.device)
        current_patience = 0

        for epoch in range(self.start, self.num_epochs):

            print("\nEpoch {}/{}".format(epoch+1, self.num_epochs))

            curr_lr                 = float(self.optimizer.param_groups[0]['lr'])
            train_cls_acc, train_loss = train_epoch(model, train_loader, self.optimizer, self.criterion, self.scheduler, self.scaler, self.device, log_batch, log_freq)
            valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, self.criterion, self.device)

            metrics = {
            'train_cls_acc': train_cls_acc,
            'train_loss': train_loss,
            }

            metrics.update({
                'valid_cls_acc': valid_cls_acc,
                'valid_loss': valid_loss,
            })

            print("\tTrain Cls. Acc {:.04f}%\tTrain Cls. Loss {:.04f}\t Learning Rate {:.07f}".format(train_cls_acc, train_loss, curr_lr))
            print("\tVal Cls. Acc {:.04f}%\tVal Cls. Loss {:.04f}".format(valid_cls_acc, valid_loss))

            # retrieval validation
            valid_ret_acc = valid_epoch_ver(model, ver_loader, self.device)
            print("Val Ret. Acc {:.04f}%".format(valid_ret_acc))
            metrics.update({
            'valid_ret_acc': valid_ret_acc
            })
            
            PBG.pbTracker.addExtraScore(train_cls_acc, 'Train')
            # restructured: boolean to add dendrite 
            model, improved, restructured, trainingComplete = PBG.pbTracker.addValidationScore(valid_cls_acc, 
            model, # .module if its a dataParallel
            "pb")
            model.to("cuda")
            if(trainingComplete):
                break
            elif(restructured):
                optimArgs = {'params':model.parameters(),'lr':model.lr}
                schedArgs = {
                "max_lr":model.lr, 
                    "total_steps" : model.total_steps, 
                    "pct_start" : 0.1, # 0.15
                    "anneal_strategy" : "cos", 
                    "final_div_factor" : 1
                } #Make sure this is lower than epochs to switch

                optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimArgs, schedArgs)

            '''
            # Logging and Early Stopping
            if log_epoch:
                ## Log metrics at each epoch in the run
                wandb.log(metrics)
            
            if early_stopping or checkpoints:
                os.makedirs(f'../autodl-fs/checkpoints/{self.model_name}', exist_ok=True) 

            if checkpoints:
                    if (epoch+1) % self.save_every == 0:
                        checkpoint_path = f"../autodl-fs/checkpoints/{self.model_name}/model_epoch_{epoch+1}.pth"
                        
                        checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                            'best_val_acc': self.best_val_acc
                        }

                        torch.save(checkpoint, checkpoint_path)
            
            if early_stopping:
                if valid_ret_acc > self.best_val_acc:
                    self.best_val_acc = valid_ret_acc
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
                        torch.save(checkpoint, f'../autodl-fs/checkpoints/{self.model_name}/best_model.pth') 
                else:
                    current_patience += 1

                if current_patience >= self.patience:
                    print(f"Best epoch was {self.best_epoch} with val_acc={self.best_val_acc*100:.2f}%.")
                    break 
                '''
