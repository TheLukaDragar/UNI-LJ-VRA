"""
Author: HanChen
Date: 21.06.2022
"""
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import random
import argparse
import json

import numpy as np
import pandas as pd
import torchvision.models as models

from transforms import build_transforms
from metrics import get_metrics
from data_utils import images_Dataloader_all
from network.models import get_swin_transformers
from network.models import get_efficientnet_ns
from network.models import MyModel, MyModelSimpl, MyModelVRA, MyModelVRA0, MyModelVRA1

import print_ERR_from_csv as print_ERR
from tqdm import tqdm

import os
import scipy.stats
from sklearn.metrics import mean_squared_error

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluating network')
           
    parser.add_argument('--root_path', default='/media/borutb/disk1/DataBase/DeepFake/DFGC-VRA/test_set3',
    #parser.add_argument('--root_path', default='/media/borutb/disk1/DataBase/DeepFake/DFGC-2022/DetectionDataset/10-frames-Public-faces',
    #parser.add_argument('--root_path', default='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/10-frames-Public-faces',
    #parser.add_argument('--root_path', default='/ceph/hpc/data/st2207-pgp-users/DataBase/DeepFake/DFGC-2022/Detection Dataset/10-frames-Private-faces',
                        type=str, help='path to Evaluating dataset')
                        
    parser.add_argument('--save_path', type=str, default='./save_result/txt2')           

    parser.add_argument('--model_name1', type=str, default='tf_efficientnet_b4_ns')           
    parser.add_argument('--model_name2', type=str, default='swin_large_patch4_window12_384_in22k')        
    parser.add_argument('--model_name', type=str, default='net1')            

    #parser.add_argument('--pre_trained', type=str, default='./save_result/models/efficientnet-b4.pth')
    parser.add_argument('--pre_trained', type=str, default='./models/swin_large_patch4_window12_384_in22k_40.pth')
    #parser.add_argument('--pre_trained_dir', type=str, default='/media/borutb/disk11/models/save_result_swin6/models/')
    parser.add_argument('--pre_trained_dir', type=str, default='save_result_net1_ARNES/models')
    #parser.add_argument('--output_txt', type=str, default='./save_result/pred_efn.txt')    
    parser.add_argument('--output_txt', type=str, default='./save_result/txt5/Test3_preds.txt')
    #parser.add_argument('--output_txt', type=str, default='./save_result/pred_swin6_private.txt')
    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_name', type=list, 
                        default=['real', 'fake'])
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--adjust_lr_iteration', type=int, default=500)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--resolution', type=int, default=384)
    parser.add_argument('--val_batch_size', type=int, default=32)

    parser.add_argument('--model', type=str, default='VRA') #[Simpl]

    args = parser.parse_args()
    return args



def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network


def test_model(model0, model1, dataloaders):
    prediction = np.array([])
    model0.train(False)
    for images in dataloaders:
        input_images = Variable(images.cuda())
        outputs1, outputs2 = model0(input_images)
        feats1_mean=outputs1.mean(dim=0)
        feats1_std=outputs1.std(unbiased=True, dim=0)
        feats2_mean=outputs2.mean(dim=0)
        feats2_std=outputs2.std(unbiased=True, dim=0)

        feats1 = torch.cat((feats1_mean, feats1_std))
        feats2 = torch.cat((feats2_mean, feats2_std))
        feats = torch.cat((feats1, feats2))
        outputs = model1(feats)

        #pred_ = torch.sigmoid(outputs)
        pred_ = outputs
        #print(outputs)
        
        pred_ = pred_.squeeze().cpu().detach().numpy()
        #pred_ = (pred_ * 4 ) + 1


        prediction = np.insert(prediction, 0, pred_)
    return prediction

def compute_metrics(y_pred, y):

  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  PLCC = scipy.stats.pearsonr(y, y_pred)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred))
  return [SRCC, PLCC, RMSE]

def main():
    args = parse_args()
    #test_videos = os.listdir(args.root_path)

    _, transform_test = build_transforms(args.resolution, args.resolution, 
                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    # create model
    model1_path = 'save_result_efficientnet_ns/models/tf_efficientnet_b4_ns.pth'
    model2_path = 'save_result_swin5/models/swin_large_patch4_window12_384_in22k_0.pth'
    model0 = MyModelVRA0(args.model_name1, model1_path, args.model_name2, model2_path, freeze=True)
    model0 = model0.cuda()
    model0.train(False)
    model0.eval()



    model1 = MyModelVRA1(1792,1536)
    model1 = model1.cuda()

    # load saved model
    trained_models = os.listdir(args.pre_trained_dir)
    #trained_models = ["swin_large_patch4_window12_384_in22k_40.pth", "swin_large_patch4_window12_384_in22k.pth"]
    path = os.path.abspath(args.pre_trained_dir)
    files = [trained_models.path for trained_models in os.scandir(path) if trained_models.is_file()]
    files.sort(key=os.path.getctime)
    for trained in [files[-1]]:
        print(trained)
        #pre_trained = os.path.join(args.pre_trained_dir, trained)
        pre_trained = trained
        model1 = load_network(model1, pre_trained).cuda()
        model1.train(False)
        model1.eval()

        
        output_txt = args.output_txt
        epoch = trained.split('_')[-1].split('.')[0]
        if epoch.isnumeric():
            output_txt = args.output_txt[:-4]+'_'+epoch+'.txt'
        result_txt = open(output_txt, 'w', encoding='utf-8')


        #load video list
        label_path = r'/media/borutb/disk1/DataBase/DeepFake/DFGC-VRA/label'
        df_test = pd.read_csv(os.path.join(label_path, 'test_set3.txt'), sep=',', names=['file'])
        names = list(df_test['file'])
        
        #set = args.root_path.split('/')[-1]
        #df_train = pd.read_csv(os.path.join(label_path, set+'.csv'), skiprows=[])
        #names = list(df_train['file'])

        #s=100
        for idx, video_name in enumerate(tqdm(names)):
            video_path = os.path.join(args.root_path, 'faces', video_name[:-4])
            test_dataset = images_Dataloader_all(video_path, transform=transform_test)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size,
                drop_last=False, shuffle=False, num_workers=0, pin_memory=False)

            with torch.no_grad():
                prediction = test_model(model0, model1, test_loader)

            if len(prediction) != 0:
                video_prediction = np.mean(prediction, axis=0)
            else:
                video_prediction = 1  # default is 1 (very bad)
            #print(video_name + '  is  fake' if np.round(video_prediction) == 1
            #    else video_name + '  is  real')
            #print('Probs %f' % video_prediction)
            video_prediction = video_prediction*4+1
            result_txt.write(video_name + ', %1.2f' % video_prediction+ '\n')
        result_txt.close()
        
        ##df_train = pd.read_csv(os.path.join(label_path, set+'.csv'), skiprows=[])
        #mos = np.array(list(df_train['mos']), dtype=float)
        
        #train_pred = pd.read_csv(output_txt, sep=',', names=['file','mos'])
        #mos_pred = np.array(list(train_pred['mos']), dtype=float)

        #mos=(mos-1)/4
        #snapshot = compute_metrics(mos_pred, mos)
        #phase=set

        
        # print('======================================================')
  
        # print('SRCC-'+phase+':', snapshot[0])
        # print('PLCC-'+phase+':', snapshot[1])
        # print('RMSE-'+phase+':', snapshot[2])
        # print('======================================================')



if __name__ == "__main__":
    args = parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main()

