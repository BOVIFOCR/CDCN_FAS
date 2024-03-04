'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' 
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020
'''

from __future__ import print_function, division
import os, sys
import torch
import matplotlib.pyplot as plt
import argparse,os
# import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datetime import datetime
import platform

from models.CDCNs import Conv2d_cd, CDCN, CDCNpp
from models.CDChannels_CNs import Conv2d_cd_channels, CDChannels_CNpp

from Load_OULUNPU_train_1RandomFrame import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPU_valtest_1RandomFrame import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from losses import contrast_depth_conv, Contrast_depth_loss
from utils import AvgrageMeter, accuracy, performances, performances_threshold
from plots import FeatureMap2Heatmap


# Dataset root
# train_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Train_images/'        
# val_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Dev_images/'     
# test_image_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/Test_images/'   

# map_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/IJCB_re/OULUtrain_images/'
# val_map_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/IJCB_re/OULUdev_images/'
# test_map_dir = '/wrk/yuzitong/DONOTREMOVE/OULU/IJCB_re/OULUtest_images/'

# train_list = '/wrk/yuzitong/DONOTREMOVE/OULU/OULU_Protocols/Protocol_1/Train.txt'
# val_list = '/wrk/yuzitong/DONOTREMOVE/OULU/OULU_Protocols/Protocol_1/Dev.txt'
# test_list =  '/wrk/yuzitong/DONOTREMOVE/OULU/OULU_Protocols/Protocol_1/Test.txt'


hostname = platform.node()

if hostname == 'duo':
    train_image_dir = '/home/rgpa18/ssan_datasets/original/oulu-npu/train/'
    val_image_dir = '/home/rgpa18/ssan_datasets/original/oulu-npu/dev/'     
    test_image_dir = '/home/rgpa18/ssan_datasets/original/oulu-npu/test/'   

    map_dir = '/home/rgpa18/ssan_datasets/original/oulu-npu/depth/train/'   
    val_map_dir = '/home/rgpa18/ssan_datasets/original/oulu-npu/depth/dev/' 
    test_map_dir = '/home/rgpa18/ssan_datasets/original/oulu-npu/depth/test/' 

    train_list = '/home/rgpa18/ssan_datasets/original/oulu-npu/Protocols/Protocol_1/Train.txt'
    val_list = '/home/rgpa18/ssan_datasets/original/oulu-npu/Protocols/Protocol_1/Dev.txt'
    test_list =  '/home/rgpa18/ssan_datasets/original/oulu-npu/Protocols/Protocol_1/Test.txt'

elif hostname == 'daugman':
    train_image_dir = '/groups/bjgbiesseck/datasets/liveness/oulu-npu/train/'
    val_image_dir = '/groups/bjgbiesseck/datasets/liveness/oulu-npu/dev/'     
    test_image_dir = '/groups/bjgbiesseck/datasets/liveness/oulu-npu/test/'   

    map_dir = '/groups/bjgbiesseck/datasets/liveness/oulu-npu/depth/train/'   
    val_map_dir = '/groups/bjgbiesseck/datasets/liveness/oulu-npu/depth/dev/' 
    test_map_dir = '/groups/bjgbiesseck/datasets/liveness/oulu-npu/depth/test/' 

    train_list = '/groups/bjgbiesseck/datasets/liveness/oulu-npu/Protocols/Protocol_1/Train.txt'
    val_list = '/groups/bjgbiesseck/datasets/liveness/oulu-npu/Protocols/Protocol_1/Dev.txt'
    test_list =  '/groups/bjgbiesseck/datasets/liveness/oulu-npu/Protocols/Protocol_1/Test.txt'

else:
    raise Exception(f'Unknown hostname \'{hostname}\'')




def train_one_epoch(args, epoch, lr, model, optimizer, criterion_absolute_loss, criterion_contrastive_loss, experiment_folder, log_file):
    loss_absolute = AvgrageMeter()
    loss_contra =  AvgrageMeter()
    #top5 = utils.AvgrageMeter()
    
    ###########################################
    '''                train                '''
    ###########################################
    model.train()
    
    # load random 16-frame clip data every epoch
    train_data = Spoofing_train(train_list, train_image_dir, map_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
    dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4)

    for i, sample_batched in enumerate(dataloader_train):
        # get the inputs
        inputs, map_label, spoof_label = sample_batched['image_x'].cuda(), sample_batched['map_x'].cuda(), sample_batched['spoofing_label'].cuda() 

        optimizer.zero_grad()
        #pdb.set_trace()
        
        # forward + backward + optimize
        map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
        
        absolute_loss = criterion_absolute_loss(map_x, map_label)
        contrastive_loss = criterion_contrastive_loss(map_x, map_label)
        
        loss =  absolute_loss + contrastive_loss
        #loss =  absolute_loss 
            
        loss.backward()
        optimizer.step()
        
        n = inputs.size(0)
        loss_absolute.update(absolute_loss.data, n)
        loss_contra.update(contrastive_loss.data, n)
    
        echo_batches = args.echo_batches
        if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
            # visualization
            FeatureMap2Heatmap(args, x_input, x_Block1, x_Block2, x_Block3, map_x, experiment_folder)

            # log written
            log_msg = 'epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss=%.4f, Contrastive_Depth_loss=%.4f' % (epoch+1, i+1, lr, loss_absolute.avg, loss_contra.avg)
            print(log_msg)
            # log_file.write(log_msg + '\n')
            # log_file.flush()
        # break

    # whole epoch average
    log_msg = 'epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch + 1, loss_absolute.avg, loss_contra.avg)
    print(log_msg)
    log_file.write(log_msg + '\n')
    log_file.flush()




def eval_model(split, args, epoch, model, dataloader_val, val_threshold, optimizer, criterion_absolute_loss, criterion_contrastive_loss, experiment_folder, log_file):
    # print('Evaluating train...')
    # print(f'Evaluating {split}...')
    loss_absolute = AvgrageMeter()
    loss_contra =  AvgrageMeter()
    model.eval()
    with torch.no_grad():
        # val for threshold
        # train_data = Spoofing_valtest(train_list, train_image_dir, map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        # dataloader_val = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)

        map_score_list = []

        for i, sample_batched in enumerate(dataloader_val):
            print(f'Evaluating {split}... batch: {i+1}/{len(dataloader_val)}', end='\r')
            # get the inputs
            inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
            train_maps = sample_batched['val_map_x'].cuda()   # binary map from PRNet

            optimizer.zero_grad()

            # pdb.set_trace()
            map_score = 0.0
            for frame_t in range(inputs.shape[1]):
                map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])

                map_label = train_maps[:,frame_t,:,:]
                absolute_loss = criterion_absolute_loss(map_x, map_label)
                contrastive_loss = criterion_contrastive_loss(map_x, map_label)
                loss = absolute_loss + contrastive_loss
                n = 1
                loss_absolute.update(absolute_loss.data, n)
                loss_contra.update(contrastive_loss.data, n)

                score_norm = torch.sum(map_x)/torch.sum(train_maps[:,frame_t,:,:])
                map_score += score_norm
            map_score = map_score/inputs.shape[1]

            map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))
            #pdb.set_trace()
        print('')

        # map_score_train_filename = args.log+'/'+ args.log+'_map_score_train.txt'
        map_score_train_filename = experiment_folder+'/'+ args.log + f'_map_score_{split}.txt'
        with open(map_score_train_filename, 'w') as file:
            file.writelines(map_score_list)

        print(f'Computing {split} performances...')
        if val_threshold is None:
            # val_threshold, test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_ACER_test_threshold = performances(map_score_val_filename, map_score_test_filename)
            val_threshold, train_ACC, train_APCER, train_BPCER, train_ACER = performances(map_score_train_filename)
        else:
            train_ACC, train_APCER, train_BPCER, train_ACER = performances_threshold(map_score_train_filename, val_threshold)

        # print('epoch:%d, Train: train_threshold=%.4f, train_ACC=%.4f, train_APCER=%.4f, train_BPCER=%.4f, train_ACER=%.4f' % (epoch+1, train_threshold, train_ACC, train_APCER, train_BPCER, train_ACER))
        log_msg = 'epoch:%d, Eval %s - Absolute_Depth_loss=%.4f, Contrastive_Depth_loss=%.4f, threshold=%.4f, ACC=%.4f, APCER=%.4f, BPCER=%.4f, ACER=%.4f' % (epoch+1, split.upper(), loss_absolute.avg, loss_contra.avg, val_threshold, train_ACC, train_APCER, train_BPCER, train_ACER)
        print(log_msg)
        # log_file.write('\nepoch:%d, Train: train_threshold=%.4f, train_ACC=%.4f, train_APCER=%.4f, train_BPCER=%.4f, train_ACER=%.4f' % (epoch+1, train_threshold, train_ACC, train_APCER, train_BPCER, train_ACER))
        log_file.write(log_msg + '\n')

        return val_threshold




# main function
def train_test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
    experiment_folder = os.path.dirname(__file__) + '/logs/' + args.log + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    isExists = os.path.exists(experiment_folder)
    if not isExists:
        os.makedirs(experiment_folder)
    log_file_name = experiment_folder + '/' + args.log+'_log_P1.txt'
    log_file = open(log_file_name, 'w')
    
    echo_batches = args.echo_batches

    log_msg = 'Oulu-NPU, P1:'
    print(log_msg)
    log_file.write(log_msg+'\n')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')
        log_file.write('finetune!\n')
        log_file.flush()
            
        model = CDCN()
        #model = model.cuda()
        model = model.to(device[0])
        model = nn.DataParallel(model, device_ids=device, output_device=device[0])
        model.load_state_dict(torch.load('xxx.pkl'))

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        
    else:
        log_msg = 'train from scratch!\n'
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()

        # model = CDCN(basic_conv=Conv2d_cd, theta=0.7)
        # model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)
        model = CDChannels_CNpp(basic_conv=Conv2d_cd_channels, theta=0.7)

        model = model.cuda()
        #model = model.to(device[0])
        #model = nn.DataParallel(model, device_ids=device, output_device=device[0])

        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # print(model) 
    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    
    #bandpass_filter_numpy = build_bandpass_filter_numpy(30, 30)  # fs, order  # 61, 64 
    ACER_save = 1.0
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        train_one_epoch(args, epoch, lr, model, optimizer, criterion_absolute_loss, criterion_contrastive_loss, experiment_folder, log_file)

        ###########################################
        '''           evaluate train            '''
        ###########################################
        train_data = Spoofing_valtest(train_list, train_image_dir, map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)
        _ = eval_model('train', args, epoch, model, dataloader_val, None, optimizer, criterion_absolute_loss, criterion_contrastive_loss, experiment_folder, log_file)

        ###########################################
        '''                  val                '''
        ###########################################
        val_data = Spoofing_valtest(val_list, val_image_dir, val_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)
        val_threshold = eval_model('val', args, epoch, model, dataloader_val, None, optimizer, criterion_absolute_loss, criterion_contrastive_loss, experiment_folder, log_file)

        ###########################################
        '''                 test                '''
        ###########################################
        test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
        _ = eval_model('test', args, epoch, model, dataloader_test, val_threshold, optimizer, criterion_absolute_loss, criterion_contrastive_loss, experiment_folder, log_file)

        log_msg = '--------------------------'
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()

    print('Finished Training')
    log_file.close()
  

  
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")

    # parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--gpu', type=int, default=0, help='the gpu id used for predict')

    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=8, help='initial batchsize')  
    parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500 
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=1400, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDChannels_CNpp_P1_1RandomFrame", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()
