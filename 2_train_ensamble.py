
import os
from train_convnext_2 import ConvNeXt
from train_swin_2 import Swin

from sklearn.linear_model import BayesianRidge
from dataset_1 import FaceFramesDataset, FaceFramesPredictionDataset, RandomFaceFramesDataset
from dataset_1 import FaceFramesDataset, build_transforms
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
#WandbLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

import torch.nn.functional as F
from timm import create_model
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import random_split
import wandb

def ensemble_predict(models, x):
    predictions = [model(x) for model in models]
    ensemble_prediction = torch.mean(torch.stack(predictions), dim=0)
    return ensemble_prediction

def train_val_test_split(dataset, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    assert 0 <= train_prop <= 1 and 0 <= val_prop <= 1 and 0 <= test_prop <= 1, "Proportions must be between 0 and 1"
    assert round(train_prop + val_prop + test_prop, 10) == 1, "Proportions must sum to 1"

    total_length = len(dataset)
    train_length = int(train_prop * total_length)
    val_length = int(val_prop * total_length)
    test_length = total_length - train_length - val_length

    return random_split(dataset, [train_length, val_length, test_length])


class Ensamble(pl.LightningModule):
    def __init__(self, 
                 modelA_hparams, modelB_hparams,
                 modelA_params = None, modelB_params = None):
        super(Ensamble, self).__init__()
        self.modelA = ConvNeXt(**modelA_hparams)
        self.modelB = Swin(**modelB_hparams)

        if modelA_params:
            self.modelA.load_state_dict(modelA_params)
        if modelB_params:
            self.modelB.load_state_dict(modelB_params)

        # self.modelA.freeze()
        # self.modelB.freeze()
        self.classifier = torch.nn.Linear(2, 1)
        self.mse = nn.MSELoss()

        self.test_preds = []
        self.test_labels = []

        # Initialize PearsonCorrCoef metric
        self.pearson_corr_coef = torchmetrics.PearsonCorrCoef().to(self.device)
        self.spearman_corr_coef = torchmetrics.SpearmanCorrCoef().to(self.device)
        #rmse
        self.mse_log = torchmetrics.MeanSquaredError().to(self.device)


        self.save_hyperparameters(ignore = ["modelA_params", "modelB_params"])

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)

        return x
    
    def RMSE(self, preds, y):
        mse = self.mse(preds.view(-1), y.view(-1))
        return torch.sqrt(mse)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.RMSE(preds, y)
        self.log('train_loss', loss.item(), on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.RMSE(preds, y)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
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

        
        self.log('test_plcc', plcc)
        self.log('test_spearman', spearman)
        self.log('test_rmse', rmse)


dataset_root = '/d/hpc/projects/FRI/ldragar/dataset'
labels_file = '/d/hpc/projects/FRI/ldragar/label/train_set.csv'
batch_size = 8
random_face_frames = True



transform_train, transform_test = build_transforms(384, 384, 
                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

print("loading dataset")



if random_face_frames:

    face_frames_dataset = RandomFaceFramesDataset(dataset_root, labels_file, transform=transform_train)
else:
    face_frames_dataset = FaceFramesDataset(dataset_root, labels_file, transform=transform_train)

train_ds, val_ds, test_ds = train_val_test_split(face_frames_dataset)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"loaded {len(train_dl)} train batches and {len(val_dl)} val batches of size {train_dl.batch_size}")

#print first train example

for x,y in train_dl:
    print(x.shape)
    print(y.shape)
    print(y)
    break

wandb_logger = WandbLogger(project='luka_borut', name='ensamble', save_dir='/d/hpc/projects/FRI/ldragar/wandb/')

modelA_hparams = {
    'model_name': 'convnext_xlarge_384_in22ft1k',
    'dropout': 0.1,
}
modelB_hparams = {
    'model_name': 'swin_large_patch4_window12_384_in22k',
    'dropout': 0.1,
}


ensamble = Ensamble(modelA_hparams, modelB_hparams)


#log
#API KEY 242d18971b1b7df61cceaa43724c3b1e6c17c49c
wandb_logger.watch(ensamble, log='all', log_freq=100)
#log batch size
wandb_logger.log_hyperparams({'batch_size': batch_size})
#random face frames
wandb_logger.log_hyperparams({'random_face_frames': random_face_frames})




#save hyperparameters
wandb_logger.log_hyperparams(modelA_hparams)
wandb_logger.log_hyperparams(modelB_hparams)


checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                      dirpath='/d/hpc/projects/FRI/ldragar/checkpoints/', 
                                      filename='ensamble-{epoch:02d}-{val_loss:.2f}', mode='min', save_top_k=1)



trainer = pl.Trainer(accelerator='gpu', 
                     max_epochs=100, 
                       log_every_n_steps=200,
                     callbacks=[
                         EarlyStopping(monitor="val_loss", 
                                       mode="min",
                                       patience=10,
                                      ),
                            checkpoint_callback

                                      
                        ]
                        ,logger=wandb_logger,
                        devices=2
                        

                    )

# Train the model
trainer.fit(ensamble, train_dl, val_dl)
# Test the model
trainer.test(ensamble, test_dl)


# #load from checkpoint

# # swin_checkpoint = '/d/hpc/projects/FRI/ldragar/checkpoints/swin_large_patch4_window12_384_in22k_40.ckpt'
# # convnext_checkpoint = '/d/hpc/projects/FRI/ldragar/checkpoints/convnext_xlarge_384_in22ft1k_40.ckpt'

# # swin_model = PawsModel.load_from_checkpoint(swin_checkpoint)
# # convnext_model = PawsModel.load_from_checkpoint(convnext_checkpoint)

# #test model
# swin_model.eval()
# convnext_model.eval()

# swin_model = swin_model.to('cuda:0')
# convnext_model = convnext_model.to('cuda:1')

# val_preds_swin = []
# val_preds_convnext = []
# val_ground_truth = []

# for x, y in val_dl:
#     x = x.to('cuda')
#     with torch.no_grad():
#         preds_swin = swin_model(x)
#         preds_convnext = convnext_model(x)
    
#     val_preds_swin.append(preds_swin.cpu().numpy())
#     val_preds_convnext.append(preds_convnext.cpu().numpy())
#     val_ground_truth.append(y.numpy())

# val_preds_swin = np.concatenate(val_preds_swin)
# val_preds_convnext = np.concatenate(val_preds_convnext)
# val_ground_truth = np.concatenate(val_ground_truth)

# # Step 2: Train the Bayesian Ridge Regression model
# X = np.column_stack((val_preds_swin, val_preds_convnext))
# y = val_ground_truth

# bayesian_ridge = BayesianRidge()
# bayesian_ridge.fit(X, y)

# test_preds_swin = []
# test_preds_convnext = []
# test_ground_truth = []

# for x,y in test_dl:
#     x = x.to('cuda')
#     with torch.no_grad():
#         preds_swin = swin_model(x)
#         preds_convnext = convnext_model(x)
    
#     test_preds_swin.append(preds_swin.cpu().numpy())
#     test_preds_convnext.append(preds_convnext.cpu().numpy())
#     test_ground_truth.append(y.numpy())

# test_preds_swin = np.concatenate(test_preds_swin)
# test_preds_convnext = np.concatenate(test_preds_convnext)

# X_test = np.column_stack((test_preds_swin, test_preds_convnext))
# ensemble_preds = bayesian_ridge.predict(X_test)

# #calculate metrics
# plcc = torchmetrics.PearsonCorrCoef(ensemble_preds, test_ground_truth)[0]
# spearman = torchmetrics.SpearmanCorrCoef(ensemble_preds, test_ground_truth)[0]
# rmse = np.sqrt(torchmetrics.MeanSquaredError(ensemble_preds, test_ground_truth))

# print(f"plcc: {plcc}, spearman: {spearman}, rmse: {rmse}")









test_labels_dir = '/d/hpc/projects/FRI/ldragar/label/'
stages = ['1','2','3']

resultsdir = './results/'
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)


for stage in stages:
    name='test_set'+stage+'.txt'
    test_labels = []
    test_names = []

    ds = FaceFramesPredictionDataset(os.path.join(test_labels_dir, name), dataset_root=dataset_root,transform=transform_test)
    print(f"loaded {len(ds)} test examples")

    with torch.no_grad():
        for x,nameee in ds:
            x = x.unsqueeze(0).to(model.device)
            y = model(x)
            y = y.cpu().numpy()
            y =y[0][0]
            
            test_labels.append(y)
            test_names.append(nameee)

    print(f"predicted {len(test_labels)} labels for {name}")
    print(f'len test_names {len(test_names)}')
    print(f'len test_labels {len(test_labels)}')





    #save to file with  Test1_preds.txt, Test2_preds.txt, Test3_preds.txt
    #name, label
    with open(os.path.join(resultsdir, 'Test'+stage+'_preds.txt'), 'w') as f:
        for i in range(len(test_names)):
            f.write(f"{test_names[i]},{test_labels[i]}\n")
        
    print(f"saved {len(test_labels)} predictions to {os.path.join(resultsdir, 'Test'+stage+'_preds.txt')}")

print("done")
        


        

    

    

# test_labels = np.concatenate(test_labels)
# print(test_labels.shape)

# #SAVE PREDICTIONS as txt file
# np.savetxt('test_labels.txt', test_labels, delimiter=',')





