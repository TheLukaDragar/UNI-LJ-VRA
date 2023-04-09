
import os
from dataset_1 import FaceFramesDataset, FaceFramesPredictionDataset, RandomFaceFramesDataset, VideoARCFramesDataset, VideoFramesDataset
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


from timm import create_model
from torch.utils.data import DataLoader
import numpy as np
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
    def __init__(self, model_name='convnext_tiny', dropout=0.1, lstm_hidden_size=512, num_layers=1):
        super(ConvNeXt, self).__init__()
        self.model_name = model_name
        self.backbone = create_model(self.model_name, pretrained=True, num_classes = 2)
        #load from checkpoint
        #self.backbone.load_state_dict(torch.load('/d/hpc/projects/FRI/ldragar/convnext_xlarge_384_in22ft1k_10.pth'))
        n_features = self.backbone.head.fc.in_features
        output_dim = 512  # You can adjust this value to control the amount of information passed to the LSTM

        # Load the pre-trained model
        self.backbone = torch.nn.DataParallel(self.backbone)
        self.backbone.load_state_dict(torch.load('/d/hpc/projects/FRI/ldragar/convnext_xlarge_384_in22ft1k_30.pth'))

        # Modify the final fully connected layer
        self.backbone.module.head.fc = nn.Linear(n_features, output_dim)

        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(output_dim + 512, lstm_hidden_size, num_layers, batch_first=True)  # Add 512 to input_dim
        self.fc = nn.Linear(lstm_hidden_size, 1)

        self.mse = nn.MSELoss()
        
        
        self.test_preds = []
        self.test_labels = []

        # Initialize PearsonCorrCoef metric
        self.pearson_corr_coef = torchmetrics.PearsonCorrCoef().to(self.device)
        self.spearman_corr_coef = torchmetrics.SpearmanCorrCoef().to(self.device)
        #rmse
        self.mse_log = torchmetrics.MeanSquaredError().to(self.device)





        self.save_hyperparameters()


        
    def RMSE(self, preds, y):
        mse = self.mse(preds.view(-1), y.view(-1))
        return torch.sqrt(mse)
        
    def forward(self, sample):
        emb, x = sample
        batch_size, num_frames, _, height, width = x.size()
        x = x.view(batch_size * num_frames, 3, height, width)

        x = self.backbone(x)  # Output shape: (batch_size * num_frames, num_features)
        x = self.drop(x)
        
        x = x.view(batch_size, num_frames, -1)
        x_with_emb = torch.cat((x, emb.unsqueeze(1).expand(-1, num_frames, -1)), dim=2)  # Concatenate embeddings with features

        _, (h_n, _) = self.lstm(x_with_emb)  # h_n has shape (num_layers, batch_size, lstm_hidden_size)

        logit = self.fc(h_n[-1])  # Use the last layer's hidden state, output shape: (batch_size, 1)
        return logit
    
    
    def training_step(self, batch, batch_idx):
        *x, y = batch
        preds = self(x)
        loss = self.RMSE(preds, y)
        self.log('train_loss', loss.item(), on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        *x, y = batch
        preds = self(x)
        loss = self.RMSE(preds, y)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True),
            'monitor': 'val_loss',
        }
        return [optimizer], [lr_scheduler]
    
    def test_step(self, batch, batch_idx):
        *x, y = batch
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


        
    def get_predictions(self):
        return torch.cat(self.test_preds).numpy()
    

if __name__ == '__main__':

    dataset_root = '/d/hpc/projects/FRI/ldragar/original_dataset'
    labels_file = '/d/hpc/projects/FRI/ldragar/label/train_set.csv'
    batch_size = 4
    seq_len = 10
    hodden_size = 512
    num_layers = 1



    transform_train, transform_test = build_transforms(384, 384, 
                            max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    print("loading dataset")



   

    vid_dataset =VideoARCFramesDataset(dataset_root, labels_file,transform=transform_train,sequence_length=seq_len)
   
    train_ds, val_ds, test_ds = train_val_test_split(vid_dataset)

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

    wandb_logger = WandbLogger(project='luka_borut', name='LSTMARCconvnext_xlarge_384_in22ft1k', save_dir='/d/hpc/projects/FRI/ldragar/wandb/')

    
    #convnext_xlarge_384_in22ft1k
    model=ConvNeXt(model_name='convnext_xlarge_384_in22ft1k', dropout=0.1, lstm_hidden_size=hodden_size,num_layers=num_layers)



    #log
    #API KEY 242d18971b1b7df61cceaa43724c3b1e6c17c49c
    wandb_logger.watch(model, log='all', log_freq=100)
    #log batch size
    wandb_logger.log_hyperparams({'batch_size': batch_size})
    #random face frames
    wandb_logger.log_hyperparams({'LSTMseq_len': seq_len})

    wandb_logger.log_hyperparams({'hidden_size': hodden_size})
    wandb_logger.log_hyperparams({'num_layers': num_layers})

    #accumulate gradient
    wandb_logger.log_hyperparams({'accumulate_grad_batches': 4})

    





    #save hyperparameters
    wandb_logger.log_hyperparams(model.hparams)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                        dirpath='/d/hpc/projects/FRI/ldragar/checkpoints/', 
                                        filename='convnext_xlarge_384_in22ft1k-{epoch:02d}-{val_loss:.2f}', mode='min', save_top_k=1)

    num_nodes = 2  # or the number of nodes you are using
      # or the number of GPUs available on each node


    trainer = pl.Trainer(accelerator='gpu', strategy='ddp',
                        devices=2,
                        num_nodes=num_nodes,
                        accumulate_grad_batches=8,

                        max_epochs=100, 
                        log_every_n_steps=200,
                        callbacks=[
                            EarlyStopping(monitor="val_loss", 
                                        mode="min",
                                        patience=10,
                                        ),
                                checkpoint_callback

                                        
                            ]
                            ,logger=wandb_logger
                        


                        )

    # Train the model
    trainer.fit(model, train_dl, val_dl)
    # Test the model
    trainer.test(model, test_dl)




    exit(0)
    #PREDICTIONS
    model.eval()  # set the model to evaluation mode

    model = model.to('cuda:0')

    test_labels_dir = '/d/hpc/projects/FRI/ldragar/label/'
    stages = ['1','2','3']
    #get wandb run id
    wandb_run_id = wandb_logger.version

    

    resultsdir = os.path.join('/d/hpc/projects/FRI/ldragar/results/', wandb_run_id)
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





