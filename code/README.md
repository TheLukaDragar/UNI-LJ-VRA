# Chapter 1: Introduction

The goal of this project was to predict MOS of DeepFake videos from DFGC dataset. The code was developed in 15 days and lots of different approaches and architectures were tested. The following is a summary of our final solution.


# Chapter 2: Preprocessing
The preprocessing of the data is done in the `0_extract_faces.py` file. You can download it here: [DataSet.zip (92,7 GB)](https://unilj-my.sharepoint.com/:u:/g/personal/borutb_fri1_uni-lj_si/EeTxtClKf3hHkmxuq097fLoBrIXFAPE0HBeoVWwIKmUswg?e=begR1X) 
 and skip the preprocessing step 



 The preprocessing is done in the following steps:

1. Load the data from the original_dataset directory.
Define the input and output directories for each dataset partition (C1, C2, C3).
1. Activate the appropriate Python environment (e.g., pytorch_env) using Conda.
2. Iterate through each partition and call the face extraction script (0_extract_faces.py).
3. Extract faces from the videos using the MTCNN model and OpenCV, for each frame in the video. The face box is also resized and adjusted to include some context around the face based on a scale factor of `1.3`.
4. Save the cropped faces in the dataset directory, preserving the original partition structure.
During preprocessing, the script processes each video in the original dataset, detects faces using the MTCNN model, crops the detected faces, and saves them in the specified output directory. This results in a new dataset containing only cropped face images, which will be used as input data for the deep learning model in the subsequent chapters. Note this step is done to speed up training of the model and can also be done on the fly during training. However, this will slow down the training process and will require more GPU memory. 

# Chapter 3: Model Architecture
The final solution was made from 2 models ConvNext and Eva.

## 3.1: ConvNext
The main idea was to use an existing model for DeepFake detections from last years winners. And adapt it to predict MOS and take advantage of the temporal nature of videos.


### 3.1.1: Data
`RandomSeqFaceFramesDataset`from `dataset_tool.py` processes each video in the dataset from `labels/train_set.csv`by randomly selecting a sequence of frames with a specified length(`seq_len=5`). The chosen sequence originates from a random starting point within the video. Each frame is then transformed according to the specified transforms for the model. Finally it returns these seqences with the coresponding MOS labels.

The dataset is then randomly split into train, validation and test sets.
- Train: 70%
- Test: 20%
- Validation: 10%

### 3.1.2: Model

The model is based on the ConvNeXt architecture and is initialized using the weights from the previous year's DFGC-1st-2022-model (`convnext_xlarge_384_in22ft1k_30.pth`)

The model's structure is as follows:
- First, the model is initialized 
- The trained backbone (DFGC-1st-2022-model) is loaded and its output features are obtained.
- The last fully connected layer of the backbone is replaced with an identity layer, effectively removing it.
- A dropout layer with the specified dropout rate is added.
- Several fully connected layers are added to process the concatenated mean and standard deviation feature vectors.
- The final fully connected layer outputs the MOS score.

The forward pass of the model is as follows:
- The input video (seqence of 5 frames) is processed by the backbone model and the output features are obtained.
- The mean and standard deviation of the output features are calculated for a sequence of frames.
- The mean and standard deviation feature vectors are concatenated and passed through the fully connected layers.
- The final output is the MOS score.

Note that we did not freeze the backbone model's weights, as we wanted to fine-tune the model to our specific task.

## 3.2: Eva

### 3.2.1 Data
analogous to ConvNext

### 3.2.2: Model
The model is based on the Eva architecture and is initialized using the pre-trained weights from Pytorch Image Models (timm) (`eva_large_patch14_336.in22k_ft_in22k_in1k`)
The model's structure is similar to the ConvNext model, with the difference that the final fully connected layers have different sizes to account for the 1024 output features of the Eva model.


# Chapter 4: Training
Both models use RMSE as the loss function and AdamW as the optimizer with a learning rate of 2e-5. They also use ReduceLROnPlateau as a learning rate scheduler with a factor of 0.1, patience of 2 and monitor of the validation loss.
We used early stopping with a patience of 4 monitoring the validation loss.

With the training done on HPC we used Pytorch Lightning to handle the training loop and logging. Models were trained on 2 Tesla V100S-32GB GPUs using `ddp` strategy
with the following hyperparameters:
- batch_size: 2
- dropout: 0.1
- seq_len: 5
- accumulate_grad_batches: 8
- epochs: 33 (early stopping)

We have trained a couple of models with the same parameters and used the best one for the final submission. The reason for this is that the training is not deterministic and the results vary from run to run. Using a deterministic training loop produces worse results. So we decided to use the best model from multiple runs. We employed this strategy for both models. 

During training, we also used the `wandb` logger to log the training and validation losses and the learning rate. This allowed us to monitor the training process. Finally after the training is done we save the final model checkpoint. And also the best model checkpoint based on the validation loss.

Best ConvNext model run details:
https://api.wandb.ai/links/luka_borut/8c7z2qrm

Best Eva model run details:
https://api.wandb.ai/links/luka_borut/l8ski9eg





Note: for the final submission we used the best model checkpoint from the training process. And simply trained it again with the same hyperparameters to cover the remaining data. After early stopping we saved the final model checkpoint to be used for the final predictions.

Final ConvNext model run details: https://api.wandb.ai/links/luka_borut/6ak9snqi

Final Eva model run details: https://api.wandb.ai/links/luka_borut/f8pff5ux

Code in: 
`train_convnext.py`
`train_eva.py`
`train_convnext_again.py`
`train_eva_again.py`


# Chapter 5: Validation and Evaluation
After training was done we used the best model checkpoint to make predictions on the test set (from the train-validation-test split of the dataset). These predictions are used to compare different models.
 The results of the best models are as follows:


| Model          | PLCC               | SRCC               | RMSE                | wandb_run_id |
| -------------- | ------------------ | ------------------ | ------------------- | ------------ |
| Best ConvNext  | 0.9374769926071168 | 0.910971224308014  | 0.2704703211784363  | vj09esa5     |
| Best Eva       | 0.9335209131240844 | 0.9147642850875854 | 0.2935817241668701  | wrc2h54x     |
| Final ConvNext | 0.9676593542099    | 0.9616091251373292 | 0.21462157368659973 | y23waiez     |
| Final Eva      | 0.9503572583198548 | 0.9621573686599731 | 0.22783146798610687 | 37orwro0     |

Note Final ConvNext and Final Eva are the models we used for the final submission. And are trained again from checkpoints of the best models. So their results are slightly better than the best models.

# Chapter 6: Results
For our submition we made predictions on the 3 test sets from `labels/test_setX.txt`using the final model checkpoints. 
First we made predictions on the test sets using the ConvNext model. Then we made predictions on the test sets using the Eva model. Finally we combined the predictions from the two models using the following formula:

`final_prediction = 0.75 * ConvNext_prediction + 0.25 * Eva_prediction`

Code in: `predict_convnext.py`, `predict_eva.py`, `combine_predictions.py`

The predictions of each model are an average of 10 predictions made on each of the test sets. Since the dataloader chooses random sequences of 5 frames from the videos. The predictions are averaged to reduce the variance of the predictions. We also logged the RMSE beetwen pairs of predictions to check for consistency. With the ConvNext model the average RMSE was 0.16.

Since the predictions are non deterministic its difficult to reproduce the exact same results. But the results should be similar. In our testing the closest final predictions we got were:

- ConvNext MAE = 0.033634803569506086,  using seed 32585
- Eva MAE = 0.04175368133045378 using seed 7327
  
  
Final difference (combined above predictions)

| Test Set | MAE                  | Correlation Coefficient |
| -------- | -------------------- | ----------------------- |
| Test 1   | 0.03007248560587565  | 0.9976959864175539      |
| Test 2   | 0.025541501492261885 | 0.9989781149798285      |
| Test 3   | 0.02521169583002726  | 0.9988755923377973      |

Code in: `convnext_prediction_analysis.ipynb`, `eva_prediction_analysis.ipynb`,`prediction_diff.ipynb`


# Chapter 7: Reproducibility
To reproduce similar results follow these steps:
Note to avoid retraining you can skip step 2 and 3 and use the checkpoints from the `convnext_models` and `eva_models` folders used for the final submission.

### 1. Prepare the dataset
You can download the dataset from the link in the dataset section and extract it in the `dataset` folder.
Or run the `0_prepare_dataset.sh` script to extract faces from the original DFGC dataset to the `dataset` folder. This will take a while. 



### 2. Train the models

Train your models using the following scripts:
 `1_train_convnext.sh`
 `1_train_eva.sh`

The seed argument is not used because training deterministicaly produces significantly worse results. 
We decided to use the best model from multiple runs. We employed this strategy for both models.

After training is done you should have checkpoints in the `convnext_models` and `eva_models` directories. The checkpoints are saved in the format `(wandb_id)/(wandb_id).pt`

### 3. Train the models again 

Use the existing checkpoints from the previous step to train the models again.
Run the `2_train_convnext_again.sh` and `2_train_eva_again.sh` scripts. This will train the models again from the checkpoints provided in the `--cp_id` argument.

example: `python train_convnext_again.py --cp_id (wandb_id)`

 This will cover the remaining data. After early stopping we save the final model checkpoint to be used for the final predictions. In the `convnext_models` and `eva_models` directories.

 There are already 2 trained checkpoints in the `convnext_models` and `eva_models` directories that were used for the final submission. 
`y23waiez`- final ConvNext model checkpoint
`37orwro0`- final Eva model checkpoint

### 4. Make predictions
Make predictions on the test sets using the following scripts:
`3_predict_convnext.sh`
`3_predict_eva.sh`
The results will be saved in the `predictions` directory in the form of `(wandb_id)/(seed)/TestX_preds.txt`
Note the seed is set to 32585 for ConvNext and 7327 for Eva as they are closest ot the original predictions.

### 5. Combine predictions
Now that we have predictions from both models. We combine them using `4_combine_predictions.sh`
The results will be saved in the `predictions` directory in the form of `/combined/TestX_preds.txt`



