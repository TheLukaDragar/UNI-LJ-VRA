import os
import random
import csv
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A

from torchvision import transforms


def build_transforms(height, width, max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406],
                     norm_std=[0.229, 0.224, 0.225], **kwargs):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.E
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
        max_pixel_value (float): max pixel value
    """

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    train_transform = A.Compose([
        # A.HorizontalFlip(),
        # A.GaussNoise(p=0.1),
        # A.GaussianBlur(p=0.1),
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        #THIS IS OK 
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=max_pixel_value),
        ToTensorV2(),
    ])

    return train_transform, test_transform


class FaceFramesDataset(Dataset):
    def __init__(self, dataset_root, labels_file, transform=None):
        self.dataset_root = dataset_root
        self.transform = transform
        
        # Read the dataset labels file
        self.video_dirs, self.mos_labels = [], []
        with open(labels_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip the first row (header)
            for row in reader:
                video_dir, mos_label = row

                # Skip c2 
                # if 'C2' in video_dir:
                #     continue

                self.video_dirs.append(os.path.join(dataset_root, video_dir))
                self.mos_labels.append(float(mos_label))

    def __getitem__(self, index):
        video_dir = self.video_dirs[index]
        mos_label = self.mos_labels[index]

        if len(os.listdir(video_dir)) == 0:
            print(video_dir)

            

        # Choose a random frame from the directory
        frame_name = os.listdir(video_dir)[0] #random.choice(os.listdir(video_dir))
        frame_path = os.path.join(video_dir, frame_name)

        # Read and transform the frame image
        frame = cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.transform:
            frame = self.transform(image=frame)["image"]

        mos_label = torch.tensor(mos_label, dtype=torch.float32)

        return frame, mos_label

    def __len__(self):
        return len(self.video_dirs)

class FaceFramesPredictionDataset(Dataset):
    """
    Dataset for predicting MOS scores from a video directory.
    it will return a list of frames from the video directory.
    """

    def __init__(self, video_list_file,dataset_root, transform=None):
        self.video_list_file = video_list_file
        self.dataset_root = dataset_root
        self.transform = transform
        self.names = []

        lines = []

        with open(self.video_list_file, 'r') as f:
            lines = [line.strip() for line in f]

        # Read the list of video directories from the provided file
        self.video_dirs = [os.path.join(self.dataset_root, line) for line in lines]
        self.names = lines



    def __getitem__(self, index):
        video_dir = self.video_dirs[index]
        name = self.names[index]


        # Load all the frames in the video directory
        frames = []
        for frame_name in sorted(os.listdir(video_dir)):

            frame_path = os.path.join(video_dir, frame_name)

            
            

            # Read and transform the frame image
            frame = cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(image=frame)["image"]

            frames.append(frame)
            break

        return frames[0],name #ONLY FOR NOW

    def __len__(self):
        return len(self.video_dirs)




class VideoFramesDataset(Dataset):
    def __init__(self, dataset_root, labels_file, transform=None, sequence_length=5):
        self.dataset_root = dataset_root
        self.transform = transform
        self.sequence_length = sequence_length

        print("Loading dataset from {}...".format(dataset_root))

        # Read the dataset labels file
        self.video_files, self.mos_labels = [], []
        with open(labels_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # skip the first row (header)
            for row in reader:
                video_file, mos_label = row

                # Skip c2
                if 'C2' in video_file:
                    continue

                self.video_files.append(os.path.join(dataset_root, video_file))
                self.mos_labels.append(float(mos_label))

        print("Loaded {} videos".format(len(self.video_files)))

    def __getitem__(self, index):
        video_file = self.video_files[index]
        mos_label = self.mos_labels[index]

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Select sequence_length equally spaced frames from the video
        frame_indices = torch.linspace(0, total_frames - 1, steps=self.sequence_length).long().tolist()
        sequence = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(image=frame)["image"]
                sequence.append(frame)

            if len(sequence) == self.sequence_length:
                break

        cap.release()

        sequence_tensor = torch.stack(sequence)
        mos_label_tensor = torch.tensor(mos_label, dtype=torch.float32)

        return sequence_tensor, mos_label_tensor

    def __len__(self):
        return len(self.video_files)



if __name__ == "__main__":
    """
    Example usage of the FaceFramesDataset class.

    """



    dataset_root = '/d/hpc/projects/FRI/ldragar/dataset'
    labels_file = '/d/hpc/projects/FRI/ldragar/label/train_set.csv'

    transform_train, transform_test = build_transforms(384, 384, 
                            max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])


    face_frames_dataset = FaceFramesDataset(dataset_root, labels_file, transform=transform_train)
    face_frames_loader = DataLoader(face_frames_dataset, batch_size=16, shuffle=True, num_workers=4)


    print(len(face_frames_dataset))

    import timm

    model_name = "convnext_xlarge_384_in22ft1k"

    # Load a pretrained model
    model = timm.create_model(model_name, pretrained=True)



    for i, (frames, mos_labels) in enumerate(face_frames_loader):
        if i == 0:
            print(frames.shape)
            print(mos_labels.shape)

            min = torch.min(frames)
            max = torch.max(frames)
            mean = torch.mean(frames)
            std = torch.std(frames)

            print(min, max, mean, std)

            #predict


            preds=model(frames).topk(5)
            print(preds)
            #tensor(-2.1179) tensor(2.6400) tensor(-0.2454) tensor(1.0441)
            #from example 
            

        break




