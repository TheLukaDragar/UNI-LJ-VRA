
print('hello')
import os
import warnings

import numpy as np
print("importing modules")
from dataset_1 import  FaceFramesSeqPredictionDataset, RandomSeqFaceFramesDataset
from dataset_1 import  build_transforms
print('imported dataset_1')
import torch
import torch.nn as nn
print('imported torch')
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
print('imported pytorch_lightning')
#WandbLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
print('imported pytorch_lightning callbacks')


from timm import create_model
print('imported timm')
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
import wandb


def train_val_test_split(dataset, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    assert 0 <= train_prop <= 1 and 0 <= val_prop <= 1 and 0 <= test_prop <= 1, "Proportions must be between 0 and 1"
    assert round(train_prop + val_prop + test_prop, 10) == 1, "Proportions must sum to 1"

    total_length = len(dataset)
    train_length = int(train_prop * total_length)
    val_length = int(val_prop * total_length)
    test_length = total_length - train_length - val_length

    return random_split(dataset, [train_length, val_length, test_length])



class ConvNeXt(pl.LightningModule):
    def __init__(self, model_name='eva_large_patch14_336.in22k_ft_in22k_in1k', dropout=0.1):
        super(ConvNeXt, self).__init__()
        self.model_name = model_name
        self.backbone = create_model(self.model_name, pretrained=True, num_classes = 0)
        #load from checkpoint
        #self.backbone.load_state_dict(torch.load('/d/hpc/projects/FRI/ldragar/convnext_xlarge_384_in22ft1k_10.pth'))
    
        n_features = 1024
     

        self.drop = nn.Dropout(dropout)
        #self.fc = nn.Linear(self.backbone.num_features, 1)
        print('model created with head feats of ', n_features + n_features)

        #average and std feature vector
        self.fc = nn.Linear(n_features+n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.mse = nn.MSELoss()
        
        
        self.test_preds = []
        self.test_labels = []


        # Initialize PearsonCorrCoef metric
        self.pearson_corr_coef = torchmetrics.PearsonCorrCoef().to(self.device)
        self.spearman_corr_coef = torchmetrics.SpearmanCorrCoef().to(self.device)
        #rmse
        self.mse_log = torchmetrics.MeanSquaredError().to(self.device)


        #ddp debug 
        
        self.seen_samples = set()






        self.save_hyperparameters()


        
    def RMSE(self, preds, y):
        mse = self.mse(preds.view(-1), y.view(-1))
        return torch.sqrt(mse)
        
    def forward(self, x):
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(batch_size * sequence_length, channels, height, width)

        features = self.backbone(x)  # Output shape: (batch_size * sequence_length, n_features)

        # Reshape to (batch_size, sequence_length, n_features)
        features = features.view(batch_size, sequence_length, -1)

        # Compute mean and standard deviation along the sequence dimension
        mean_features = torch.mean(features, dim=1)
        std_features = torch.std(features, dim=1)

        # Concatenate mean and standard deviation feature vectors
        concat_features = torch.cat((mean_features, std_features), dim=1)
        x = self.drop(concat_features)
        x = torch.nn.functional.relu(self.fc(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        logit = x

        return logit
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.RMSE(preds, y)
        self.log('train_loss', loss.item(), on_epoch=True, prog_bar=True,sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.RMSE(preds, y)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True,sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True),
            'monitor': 'val_loss',
        }
        return [optimizer], [lr_scheduler]
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        #Check for duplicates THIS IS FOR DDP DEBUGGING
        for sample in x:
            sample_hash = hash(sample.cpu().numpy().tostring())
            if sample_hash in self.seen_samples:
                print("Duplicate sample detected:", sample)
                warnings.warn("Duplicate sample detected!!! ")
            else:
                print("New sample detected:", str(sample_hash),str(batch_idx))
                self.seen_samples.add(sample_hash)
        preds = self(x)
        self.test_preds.append(preds)
        self.test_labels.append(y)
        

    def on_test_epoch_end(self):
        test_preds = torch.cat(self.test_preds)
        test_labels = torch.cat(self.test_labels)
        test_preds = test_preds.view(-1)
        plcc = self.pearson_corr_coef(test_preds, test_labels)
        spearman = self.spearman_corr_coef(test_preds, test_labels)
        mse_log = self.mse_log(test_preds, test_labels)
        rmse = torch.sqrt(mse_log)

        
        self.log('test_plcc', plcc,sync_dist=True)
        self.log('test_spearman', spearman,sync_dist=True)
        self.log('test_rmse', rmse,sync_dist=True)


        
    def get_predictions(self):
        return torch.cat(self.test_preds).numpy()
    

if __name__ == '__main__':

    print("starting")

    dataset_root = '/d/hpc/projects/FRI/ldragar/dataset'
    labels_file = '/d/hpc/projects/FRI/ldragar/label/train_set.csv'
    batch_size = 2
   


    transform_train, transform_test = build_transforms(336, 336, 
                            max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    print("loading dataset")



   

   
   


    

    #convnext_xlarge_384_in22ft1k
    model=ConvNeXt(model_name='eva_large_patch14_336.in22k_ft_in22k_in1k', dropout=0.1)
                

   

    wandb_run_id = 'ka8kp1oy' #first fema very good
    checkpoint_root = '/d/hpc/projects/FRI/ldragar/models'
    seq_len = 10 #important for the model
    x_predictions = 10 #how many predictions to average

    res_name="10f_ema"+wandb_run_id+"_final_modl_"+str(x_predictions)+"_avgpreds"



    #load checkpoint
    #get files in dir
    files = os.listdir(checkpoint_root)
    #get the one with the same run id
    cp_name = [f for f in files if wandb_run_id in f][0]
    print(cp_name)
    if not cp_name.endswith('.ckpt'):
        #this is a pt file 
        cp_name = os.path.join(checkpoint_root, cp_name,cp_name+'.pt')
    

    #load checkpoint
    if cp_name.endswith('.ckpt'):

        #load checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_root, cp_name))
        model.load_state_dict(checkpoint['state_dict'])

    else:
        #load pt file
        model.load_state_dict(torch.load(cp_name))

    
    



    #PREDICTIONS
    model.eval()  # set the model to evaluation mode

    model = model.to('cuda:0')

    test_labels_dir = '/d/hpc/projects/FRI/ldragar/label/'
    stages = ['1','2','3']
    #get wandb run id


    resultsdir = os.path.join('./resFinalt', res_name)
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)


for stage in stages:
    name='test_set'+stage+'.txt'

    # Initialize lists to store the predictions and the test names
    all_test_labels = []
    all_test_names = []

    # Make x_predictions for each stage
    for i in range(x_predictions):
        test_labels = []
        test_names = []

        ds = FaceFramesSeqPredictionDataset(os.path.join(test_labels_dir, name), dataset_root, transform=transform_test, seq_len=seq_len)
        print(f"loaded {len(ds)} test examples")

        with torch.no_grad():
            for x, nameee in ds:
                x = x.unsqueeze(0).to(model.device)
                y = model(x)
                y = y.cpu().numpy()
                y = y[0][0]

                test_labels.append(y)
                test_names.append(nameee)

        print(f"predicted {len(test_labels)} labels for {name}")

        all_test_labels.append(test_labels)
        all_test_names.append(test_names)

    # Calculate the mean and standard deviation of the predictions
    all_test_labels = np.array(all_test_labels)
    mean_test_labels = np.mean(all_test_labels, axis=0)
    std_test_labels = np.std(all_test_labels, axis=0)

    # Calculate the RMSE between each pair of predictions
    rmse_list = []
    for i in range(x_predictions):
        for j in range(i+1, x_predictions):
            rmse = np.sqrt(np.mean((all_test_labels[i] - all_test_labels[j])**2))
            rmse_list.append(rmse)

    print(f"RMSE between predictions: {rmse_list}")

    # Save the mean predictions to a file
    with open(os.path.join(resultsdir, 'Test'+stage+'_preds.txt'), 'w') as f:
        for i in range(len(all_test_names[0])):
            f.write(f"{all_test_names[0][i]},{mean_test_labels[i]}\n")

    print(f"saved {len(mean_test_labels)} predictions to {os.path.join(resultsdir, 'Test'+stage+'_preds.txt')}")

print("done")

