# %% [code]
import torch

# Utility function for saving model
# During training, the loss values are stored in a list.
# We check the last two values to see if the loss has reduced.

class ModelHistory():
    def __init__(self):
        self.best_loss = 999
        self.train_loss = []
        self.valid_loss = []
        self.valid_acc = []

    def add_metric(self, train_loss=None, valid_loss=None, valid_acc=None):
        if train_loss!=None:
            self.train_loss.append(train_loss)
        if valid_loss != None:
            self.valid_loss.append(valid_loss)
        if valid_acc != None:
            self.valid_acc.append(valid_acc)

    def get_metrics(self):
        data = {"train loss":self.train_loss,"valid loss:":self.valid_loss,"valid accuracy":self.valid_acc}
        return data

    def save_checkpoint(self, state, loss_value, mode="valid", name="min_loss_state", unconditional=False):
        if not unconditional:
            if mode=="valid":
                self.best_loss = self.best_loss if len(self.valid_loss)==0 else min(self.valid_loss)
                if self.best_loss>=loss_value:        
                    print("    Loss reduced by:\t", round((self.best_loss - loss_value),4), ". Saving model state.")
                    torch.save(state, name + ".dct")
                    self.best_loss = loss_value

            if mode=="train":
                self.best_loss = self.best_loss if len(self.train_loss)==0 else min(self.train_loss)
                if self.best_loss>=loss_value:        
                    print("    Loss reduced by:\t", round((self.best_loss - loss_value),4), ". Saving model state.")
                    torch.save(state, name + ".dct")
                    self.best_loss = loss_value
        else:
            torch.save(state, name + ".dct")
            print("Model saved as",str(name+".dct"))
