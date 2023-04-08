import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
import argparse
import os

#salloc --job-name "InteractiveJob" --cpus-per-task 1 --mem-per-cpu 1500 --time 10:00

print(torch.cuda.is_available())


#run 

salloc --job-name=Interactive_GPU --partition=gpu --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem-per-gpu=64G --time=2:00:00

