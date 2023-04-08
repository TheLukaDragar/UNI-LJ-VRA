"""
Author: Borut Batagelj
Date: 25.09.2022

Combine EFFICIENTNET-NS + SWIN
"""

# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
import argparse

import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from logger import create_logger

from network.models import get_swin_transformers
from network.models import get_efficientnet_ns
from network.models import MyModel, MyModelSimpl, MyModelVRA, MyModelVRA0, MyModelVRA1


from transforms import build_transforms
from metrics import get_metrics
from dataset import binary_Rebalanced_Dataloader

import os
from torch.utils.data import WeightedRandomSampler


######################################################################
# Save model
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network

def parse_args():
    parser = argparse.ArgumentParser(description='Training network')
           
#    parser.add_argument('--root_path_dfdc', default='/data/linyz/DFDC/face_crop_png',
#                        type=str, help='path to DFDC dataset')         
    parser.add_argument('--root_path_dfdc', default='/media/borutb/Seagate Backup Plus Drive/Database/DFDC/kaggle/train_sample_videos1_face_crop_png/',
                        type=str, help='path to DFDC dataset')    
    parser.add_argument('--root_path', default='/d/hpc/projects/FRI/borutb/DataBase/DeepFake/',
                        type=str, help='path to datasets')    
    parser.add_argument('--save_path', type=str, default='./save_result_net1_02')           

    parser.add_argument('--model_name1', type=str, default='tf_efficientnet_b4_ns')           
    parser.add_argument('--model_name2', type=str, default='swin_large_patch4_window12_384_in22k')        
    parser.add_argument('--model_name', type=str, default='net1_vra2')            

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_name', type=list, 
                        default=['real', 'fake'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--adjust_lr_iteration', type=int, default=100)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--val_batch_size', type=int, default=128)
    args = parser.parse_args()
    return args

def load_txt(txt_path='./txt', logger=None):
    txt_names = os.listdir(txt_path)
    all_videos, all_labels = [], []
    for txt_name in txt_names:
        tmp_videos, tmp_labels = [], []
        with open(os.path.join(txt_path, txt_name), 'r') as f:
            videos_names = f.readlines()
            for i in videos_names:
                i = os.path.join(args.root_path, i)
                if i.find('landmarks') != -1:
                    continue
                if len(os.listdir(i.strip().split()[0])) == 0:
                    continue
                tmp_videos.append(i.strip().split()[0])
                tmp_labels.append(int(i.strip().split()[1]))

                all_videos.append(i.strip().split()[0])
                all_labels.append(int(i.strip().split()[1]))

        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        videos = len(tmp_labels)
        fake = sum(tmp_labels)
        real = videos - fake

        allvideos = len(all_labels)
        allfake = sum(all_labels)
        allreal = allvideos - allfake

        print('{} Videos:{:5d} Real:{:5d} Fake:{:5d} Ratio:{:5.2f}       All Videos:{:5d} All Real:{:5d} All Fake:{:5d} All Ratio:{:5.2f}   Dataset: {}'
        .format(timeStr, videos, real, fake, fake/real, allvideos, allreal, allfake, allfake/allreal, txt_name))    
    return all_videos, all_labels

def load_csv(txt_path='./csv', logger=None):
    txt_names = os.listdir(txt_path)
    all_videos, all_labels = [], []
    for txt_name in txt_names:
        tmp_videos, tmp_labels = [], []
        with open(os.path.join(txt_path, txt_name), 'r') as f:
            videos_names = f.readlines()
            for i in videos_names:
                i = os.path.join(args.root_path, i)
                if i.find('landmarks') != -1:
                    continue
                if len(os.listdir(i.strip().split()[0])) == 0:
                    continue
                tmp_videos.append(i.strip().split()[0])
                tmp_labels.append(float(i.strip().split()[1]))

                all_videos.append(i.strip().split()[0])
                all_labels.append(float(i.strip().split()[1]))

        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        videos = len(tmp_labels)
        mos = float(sum(tmp_labels)/len(tmp_labels))

        print('{} Videos:{:5d} Mean mos:{:f} Dataset: {}'
        .format(timeStr, videos, mos, txt_name))    
    return all_videos, all_labels


def make_weights_for_balanced_classes(train_dataset):
    targets = []

    targets = torch.tensor(train_dataset)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight

def main():
    args = parse_args()
    print(args.base_lr)

    logger = create_logger(output_dir='%s/report' % args.save_path, name=f"{args.model_name1+' and '+args.model_name2}")
    logger.info('Start Training %s' % args.model_name1 + '+' + args.model_name2)
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    logger.info(timeStr)  

    transform_train, transform_test = build_transforms(args.resolution, args.resolution, 
                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    train_videos, train_labels = [], []
    for idx in tqdm(range(0, 0)): #for idx in tqdm(range(0, 50)):
        sub_name = 'dfdc_train_part_%d' % idx
        video_sub_path = os.path.join(args.root_path_dfdc, sub_name)
        with open(os.path.join(video_sub_path, 'metadata.json')) as metadata_json:
            metadata = json.load(metadata_json)
        for key, value in metadata.items(): 
            if value['label'] == 'FAKE': # FAKE or REAL
                label = 1
            else:
                label = 0
            inputPath = os.path.join(args.root_path_dfdc, sub_name, key)
            if not os.path.isdir(inputPath) or len(os.listdir(inputPath)) == 0: #BB or not exist
                continue
            train_videos.append(inputPath)
            train_labels.append(label)
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    if len(train_labels) != 0:
        print(timeStr, len(train_labels), sum(train_labels), sum(train_labels)/len(train_labels))
    else:
        print(timeStr, len(train_labels), sum(train_labels))
    
    tmp_videos, tmp_labels = load_csv(txt_path='./txt5')
    train_videos += tmp_videos
    train_labels += tmp_labels
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    

    dataset = binary_Rebalanced_Dataloader(video_names=train_videos, video_labels=train_labels, phase='train', 
                                                num_class=args.num_class, transform=transform_train)


    train_dataset, val_dataset = torch.utils.data.random_split(dataset=dataset,
                                lengths=[560, 140],
                                generator=torch.Generator().manual_seed(42))

    model1_path = 'save_result_efficientnet_ns/models/tf_efficientnet_b4_ns.pth'
    model2_path = 'save_result_swin5/models/swin_large_patch4_window12_384_in22k_0.pth'
    model0 = MyModelVRA0(args.model_name1, model1_path, args.model_name2, model2_path, freeze=True)
    model0 = model0.cuda()

    model1 = MyModelVRA1(1792,1536)
    model1 = model1.cuda()





    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)
    criterion = torch.nn.MSELoss()


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, drop_last=True,
                                               shuffle=True, num_workers=6, pin_memory=True)



    loss_name = ['BCE']
    iteration = 0
    running_loss = {loss: 0 for loss in loss_name}
#    real = 0
#    fake = 0
    for epoch in range(args.num_epochs):
        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        logger.info(timeStr+'Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        logger.info(timeStr+'-' * 10)

        model1.train(True)  # Set model to training mode
        # Iterate over data (including images and labels).
        for index, (images, labels) in enumerate(train_loader):
            iteration += 1

            labels = Variable(labels.cuda().detach())
            xx1=[]
            xx2=[]
            for i in range(len(images)):
                image = Variable(images[i].cuda().detach())           
                x1, x2 = model0(image)
                x1= x1.cpu().detach().numpy()
                x2= x2.cpu().detach().numpy()
                xx1.append(x1)
                xx2.append(x2)

            #feats1_mean = np.vstack(xx1).mean(0)
            mean_rows = [np.mean([arr[i] for arr in xx1], axis=0) for i in range(len(xx1[0]))]
            feats1_mean = np.stack(mean_rows)
            #feats1_std = np.vstack(xx1).std(axis=0, ddof=1)
            std_rows = [np.std([arr[i] for arr in xx1], axis=0, ddof=1) for i in range(len(xx1[0]))]
            feats1_std = np.stack(std_rows)
            feats1 = np.concatenate((feats1_mean, feats1_std), axis=1)

            
            mean_rows = [np.mean([arr[i] for arr in xx2], axis=0) for i in range(len(xx2[0]))]
            feats2_mean = np.stack(mean_rows)
            std_rows = [np.std([arr[i] for arr in xx2], axis=0, ddof=1) for i in range(len(xx2[0]))]
            feats2_std = np.stack(std_rows)
            feats2 = np.concatenate((feats2_mean, feats2_std), axis=1)

            feats = np.concatenate((feats1, feats2), axis=1)
            
            feats = torch.from_numpy(feats).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model1(feats)
            # Calculate loss
            #loss = criterion(outputs.squeeze(), (labels.float()-1)/4)
            loss = criterion(outputs.squeeze(), labels.float())


            # update the parameters
            loss.backward()
            optimizer.step()

            running_loss['BCE'] += loss.item()
            # break

#            real += sum(labels==0).item()
#            fake += sum(labels==1).item()
#            if fake != 0:
#                logger.info('Real: {}, Fake: {}, rate: {}'.format(real, fake, real/fake))
#            else:
#                logger.info('Real: {}, Fake: {}'.format(real, fake))
            

            if iteration % 10 == 0:
                timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
                logger.info(timeStr+'Epoch: {:g}, Itera: {:g}, Global Itera: {:g}, Step: {:g}, BCE: {:g} '.
                      format(epoch, index, iteration, len(train_loader), *[running_loss[name] / 10 for name in loss_name]))
                running_loss = {loss: 0 for loss in loss_name}
            
            if iteration % args.adjust_lr_iteration == 0:
                scheduler.step()
                #logger.info('now lr is : {}\n'.format(scheduler.get_last_lr()))
                
        if epoch % 10 == 0:
            timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
            logger.info(timeStr + '  Save  Model  ')
            save_network(model1, '%s/models/%s_%d.pth' % (args.save_path, args.model_name, epoch))
            
    save_network(model1, '%s/models/%s.pth' % (args.save_path, args.model_name))


if __name__ == "__main__":
    args = parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists('%s/models' % args.save_path):
        os.makedirs('%s/models' % args.save_path)
    if not os.path.exists('%s/report' % args.save_path):
        os.makedirs('%s/report' % args.save_path)

    main()
