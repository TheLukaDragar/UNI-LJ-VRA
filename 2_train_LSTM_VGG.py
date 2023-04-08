
print("Importing Libraries")
import wandb
from dataset_1 import VideoFramesDataset, build_transforms
print("Imported Libraries 1")
import torch
print("Imported torch")
import torch.nn as nn
print("Imported nn")
import pytorch_lightning as pl
print("Imported pl")
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
print("Imported EarlyStopping")
import torchmetrics
print("Imported torchmetrics")
#WandbLogger
from pytorch_lightning.loggers import WandbLogger
print("Imported WandbLogger")


from timm import create_model
print("Imported create_model")
from torch.utils.data import DataLoader
print("Imported DataLoader")
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split

from facenet_pytorch import InceptionResnetV1


def train_val_test_split(dataset, train_prop=0.7, val_prop=0.2, test_prop=0.1):
    assert 0 <= train_prop <= 1 and 0 <= val_prop <= 1 and 0 <= test_prop <= 1, "Proportions must be between 0 and 1"
    assert round(train_prop + val_prop + test_prop, 10) == 1, "Proportions must sum to 1"

    total_length = len(dataset)
    train_length = int(train_prop * total_length)
    val_length = int(val_prop * total_length)
    test_length = total_length - train_length - val_length

    return random_split(dataset, [train_length, val_length, test_length])


class VGGFaceResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGFaceResNet50, self).__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2' if pretrained else None)
        self.num_features = self.backbone.last_linear.out_features
        
    def forward(self, x):
        return self.backbone(x)

class MOSVideoSequenceModel(nn.Module):
    def __init__(self, backbone_name, lstm_hidden_size=256, lstm_num_layers=3):
        super(MOSVideoSequenceModel, self).__init__()

        # Load the pre-trained backbone model
        self.backbone = VGGFaceResNet50(pretrained=True)
        self.feature_dim = self.backbone.num_features

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Define the final fully connected layer
        self.fc = nn.Linear(lstm_hidden_size, 1)

    def forward(self, x):
        # Extract features from the backbone model
       # Extract features from the input sequence using the backbone
        features = [self.backbone(frame) for frame in x]
        features = torch.stack(features, dim=1)

        # Pass the features through the LSTM layer
        lstm_out, _ = self.lstm(features)

        # Predict the MOS score using the fully connected layer
        mos_score = self.fc(lstm_out[:, -1, :])
        return mos_score

class MOSPredictor(pl.LightningModule):
    def __init__(self, backbone_name="vggface_r50", learning_rate=1e-3):
        super(MOSPredictor, self).__init__()
        self.model = MOSVideoSequenceModel(backbone_name)
        self.learning_rate = learning_rate
        self.loss = nn.MSELoss()
        self.pearson_corr_coef = torchmetrics.PearsonCorrCoef().to(self.device)
        self.spearman_corr_coef = torchmetrics.SpearmanCorrCoef().to(self.device)
        self.mse_log = torchmetrics.MeanSquaredError().to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        lr_scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                            patience=2, verbose=True),
                                            'monitor': 'val_loss',
        }
        return [optimizer], [lr_scheduler]
    

    
    


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

        

 
dataset_root = '/d/hpc/projects/FRI/ldragar/original_dataset'
labels_file = '/d/hpc/projects/FRI/ldragar/label/train_set.csv'

transform_train, transform_test = build_transforms(384, 384, 
                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

print("Loading dataset")


face_frames_dataset =  VideoFramesDataset(dataset_root, labels_file,transform=transform_train)


print("Splitting dataset")



train_ds, val_ds, test_ds = train_val_test_split(face_frames_dataset)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

print(f"loaded {len(train_dl)} train batches and {len(val_dl)} val batches of size {train_dl.batch_size}")

#print first train example

for x,y in train_dl:
    print(x.shape)
    print(y.shape)
    print(y)
    break

print("Creating model")

wandb_logger = WandbLogger(project='luka_borut', name='vggface_r50_lstm', save_dir='/d/hpc/projects/FRI/ldragar/wandb/')


#convnext_xlarge_384_in22ft1k
model = MOSPredictor(backbone_name="vggface_r50", learning_rate=1e-3)

print("Initializing trainer")

#log
#API KEY 242d18971b1b7df61cceaa43724c3b1e6c17c49c
wandb_logger.watch(model, log='all', log_freq=100)
#save hyperparameters
wandb_logger.log_hyperparams(model.hparams)




trainer = pl.Trainer(accelerator='gpu', 
                     max_epochs=100, 
                       log_every_n_steps=100,
                     callbacks=[
                         EarlyStopping(monitor="val_loss", 
                                       mode="min",
                                       patience=2,
                                      )
                        ]
                        ,logger=wandb_logger

                    )

print("Training model")
# Train the model
trainer.fit(model, train_dl, val_dl)
# Test the model
trainer.test(model, test_dl)







exit()
#PREDICTIONS
# model.eval()  # set the model to evaluation mode



# test_labels_dir = '/d/hpc/projects/FRI/ldragar/label/'
# stages = ['1','2','3']
# test_labels = []
# for stage in stages:
#     name='test_set'+stage+'.txt'

#     ds = FaceFramesPredictionDataset(os.path.join(test_labels_dir, name), dataset_root=dataset_root,transform=transform_test)
#     with torch.no_grad():
#         for x in ds:
#             x = x.unsqueeze(0)
#             y = model(x)
#             test_labels.append(y)

# test_labels = np.concatenate(test_labels)
# print(test_labels.shape)

# #SAVE PREDICTIONS as txt file
# np.savetxt('test_labels.txt', test_labels, delimiter=',')





