# Chapter 1: Introduction


# Chapter 2: Preprocessing
The preprocessing of the data is done in the `0_extract_faces.py` file. The preprocessing is done in the following steps:

1. Load the data from the original_dataset directory.
Define the input and output directories for each dataset partition (C1, C2, C3).
2. Activate the appropriate Python environment (e.g., pytorch_env) using Conda.
3. Iterate through each partition and call the face extraction script (0_extract_faces.py).
4. Extract faces from the videos using the MTCNN model and OpenCV, for each frame in the video. Also zoom in the face ba a factor of `1.3`.
5. Save the cropped faces in the dataset directory, preserving the original partition structure.
During preprocessing, the script processes each video in the original dataset, detects faces using the MTCNN model, crops the detected faces, and saves them in the specified output directory. This results in a new dataset containing only cropped face images, which will be used as input data for the deep learning model in the subsequent chapters. Note this step is done to speed up training of the model and can also be done on the fly during training. However, this will slow down the training process and will require more GPU memory. 

# Chapter 3: Model Architecture and Design
The final solution was made from the following models:

## 3.1: ConvNext
The main idea was to use an existing model for DeepFake detections from last years winners. And adapt it to predict MOS and take advantage of the temporal nature of videos.


### 3.1.1: Data
`RandomSeqFaceFramesDataset`from `dataset_tool.py` processes each video in the dataset by randomly selecting a sequence of frames with a specified length(`seq_len=5`). The chosen sequence originates from a random starting point within the video. each frame is then transformed according to the specified transforms for the model. Finally it returns these seqences with the coresponding MOS labels.

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
The model is based on the Eva architecture and is initialized using the pre-trained weights from timm (`eva_large_patch14_336.in22k_ft_in22k_in1k`)
The model's structure is similar to the ConvNext model, with the difference that the final fully connected layers have different sizes to account for the 1024 output features of the Eva model.


# Chapter 4: Training
Both models use RMSE as the loss function and AdamW as the optimizer with a learning rate of 2e-5. They also use ReduceLROnPlateau as a learning rate scheduler with a factor of 0.1, patience of 2 and monitor of the validation loss.
We used early stopping with a patience of 4 monitoring the validation loss.

With the training done on HPC we used Pytorch Lightning to handle the training loop and logging. Models were trained on 2 Tesla V100S-32GB using `ddp` strategy.
with the following hyperparameters:
- batch_size: 2
- dropout: 0.1
- seq_len: 5
- accumulate_grad_batches: 8







# Chapter 5: Validation and Evaluation

# Chapter 6: Results and Analysis

# Chapter 7: Conclusion and Future Work
