import os, sys
import numpy as np
import cv2

import math
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import matplotlib.pyplot as plt

# from CDCNs import Conv2d_cd, CDCN, CDCNpp


global_in_channels = 3
global_out_channels = 6
global_weight_fill = 0.5
# global_weight_fill = 1




class MicroVanillaConv(nn.Module):

    def __init__(self):   
        super(MicroVanillaConv, self).__init__()

        # self.conv1 = nn.Sequential(
        #     basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),    
        # )
        self.conv1 = self.conv = nn.Conv2d(global_in_channels, global_out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.conv.weight.data.fill_(global_weight_fill)

    def forward(self, x):	    	# x [3, 256, 256]
        out = self.conv1(x)
        return out





class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.conv.weight.data.fill_(global_weight_fill)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        print('out_normal:', out_normal)
        print('out_normal.shape:', out_normal.shape)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            print('self.conv.weight:', self.conv.weight)
            print('self.conv.weight.shape:', self.conv.weight.shape)
            print('self.conv.weight.sum(2):', self.conv.weight.sum(2))
            print('self.conv.weight.sum(2).shape:', self.conv.weight.sum(2).shape)
            print('self.conv.weight.sum(2).sum(2):', self.conv.weight.sum(2).sum(2))
            print('self.conv.weight.sum(2).sum(2).shape:', self.conv.weight.sum(2).sum(2).shape)
            # sys.exit(0)
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            print('kernel_diff:', kernel_diff)
            print('kernel_diff.shape:', kernel_diff.shape)
            # sys.exit(0)
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            print('out_diff.shape:', out_diff.shape)
            sys.exit(0)
            return out_normal - self.theta * out_diff

class MicroCDCN(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7 ):   
        super(MicroCDCN, self).__init__()

        # self.conv1 = nn.Sequential(
        #     basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),    
        # )
        self.conv1 = nn.Sequential(
            basic_conv(global_in_channels, global_out_channels, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            # nn.BatchNorm2d(5),
            # nn.ReLU(),    
        )

    def forward(self, x):	    	# x [3, 256, 256]
        out = self.conv1(x)
        return out
    




class Conv2d_cd_channels(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd_channels, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv.weight.data.fill_(global_weight_fill)
        self.theta = theta

    def forward(self, x):


        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff1 = self.conv.weight.sum(2).sum(2)
            kernel_diff2 = torch.roll(kernel_diff1, shifts=1, dims=1)
            # kernel_diff3 = torch.roll(kernel_diff1, shifts=2, dims=1)

            kernel_diff1 = kernel_diff1[:, :, None, None]
            kernel_diff2 = kernel_diff2[:, :, None, None]
            # kernel_diff3 = kernel_diff3[:, :, None, None]

            out_diff1 = F.conv2d(input=x, weight=kernel_diff1, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)
            out_diff2 = F.conv2d(input=x, weight=kernel_diff2, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - (0.5*self.theta * out_diff1) - (0.5*self.theta * out_diff2)

class MicroCDCN_Channels(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd_channels, theta=0.7):   
        super(MicroCDCN_Channels, self).__init__()

        # self.conv1 = nn.Sequential(
        #     basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),    
        # )
        self.conv1 = nn.Sequential(
            basic_conv(global_in_channels, global_out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1, theta=theta),
            # nn.BatchNorm2d(5),
            # nn.ReLU(),    
        )

    def forward(self, x):	    	# x [3, 256, 256]
        out = self.conv1(x)
        return out




def plot_feature_maps_1_model(img, feature_maps, title, image_path):
    feature_maps = feature_maps.detach().numpy()
    B, C, W, H = feature_maps.shape

    fig, axes = plt.subplots(1, C+1, figsize=((C+1)*2, 2))

    axes[0].imshow(img)
    axes[0].set_title('Input')
    axes[0].axis('off')

    for i in range(1, C+1, 1):
        axes[i].imshow(feature_maps[0,i-1], cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f'Feature Map {i}')
        axes[i].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()


def plot_feature_maps_2_model(img, feature_maps1, feature_maps2, title, image_path):
    feature_maps1 = feature_maps1.detach().numpy()
    feature_maps2 = feature_maps2.detach().numpy()
    B, C, W, H = feature_maps1.shape

    rows = 3
    cols = C+1
    constrained_layout = False
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), constrained_layout=constrained_layout)

    row = 0
    for i in range(0, cols):
        axes[row,i].axis('off')
    axes[row,1].imshow(img)
    axes[row,1].set_title('Input')
    axes[row,2].imshow(img[:,:,0], cmap='gray', vmin=0, vmax=255)
    axes[row,2].set_title('R')
    axes[row,3].imshow(img[:,:,1], cmap='gray', vmin=0, vmax=255)
    axes[row,3].set_title('G')
    axes[row,4].imshow(img[:,:,2], cmap='gray', vmin=0, vmax=255)
    axes[row,4].set_title('B')
    

    row = 1
    axes[row,0].axis('off')
    axes[row,0].text(1, 0.5, 'Central Diff\nConv (CDC)')
    for i in range(1, cols):
        axes[row,i].imshow(feature_maps1[0,i-1], cmap='gray', vmin=0, vmax=255)
        axes[row,i].set_title(f'Feature Map {i}')
        axes[row,i].axis('off')

    row = 2
    axes[row,0].axis('off')
    axes[row,0].text(1, 0.5, 'Central Diff\nConv CHANNELS (CDCC)')
    for i in range(1, cols):
        axes[row,i].imshow(feature_maps2[0,i-1], cmap='gray', vmin=0, vmax=255)
        axes[row,i].set_title(f'Feature Map {i}')
        axes[row,i].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()




if __name__ == '__main__':

    input_img_path = 'models/face1.png'
    output_microVanillaConv_path = 'models/face1_output_microVanillaConv.png'
    output_microCDCN_path = 'models/face1_output_microCDCN.png'
    output_microCDCN_Channels_path = 'models/face1_output_microCDCN_Channels.png'
    output_2_models_path = 'models/face1_output_2_models.png'

    print(f'Loading \'{input_img_path}\'')
    img = cv2.imread(input_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_rgb = np.transpose(img, (2, 0, 1))
    im_rgb = im_rgb / 255.
    im_rgb = torch.from_numpy(im_rgb).float()
    im_rgb = torch.unsqueeze(im_rgb, 0)
    # print('im_rgb:', im_rgb)
    print('im_rgb.shape:', im_rgb.shape)



    # print('---------------')
    # microVanillaConv = MicroVanillaConv()
    # out_microVanillaConv = microVanillaConv(im_rgb)
    # out_microVanillaConv = (out_microVanillaConv*255).float()
    # # print('microCDCN:', dir(microCDCN))
    # for param in microVanillaConv.parameters():
    #     print('microVanillaConv:', param)
    #     print('microVanillaConv:', param.shape)
    # # print('out_microCDCN:', out_microCDCN)
    # print('out_microVanillaConv.shape:', out_microVanillaConv.shape)
    # title_out_microVanillaConv = 'Vanilla Convolution'
    # plot_feature_maps_1_model(img, out_microVanillaConv, title_out_microVanillaConv, output_microVanillaConv_path)



    # print('---------------')
    # microCDCN = MicroCDCN(basic_conv=Conv2d_cd, theta=0.7)
    # out_microCDCN = microCDCN(im_rgb)
    # out_microCDCN = (out_microCDCN*255).float()
    # # print('microCDCN:', dir(microCDCN))
    # for param in microCDCN.parameters():
    #     print('microCDCN:', param)
    #     print('microCDCN:', param.shape)
    # # print('out_microCDCN:', out_microCDCN)
    # print('out_microCDCN.shape:', out_microCDCN.shape)
    # title_out_microCDCN = 'Central Difference Convolution'
    # plot_feature_maps_1_model(img, out_microCDCN, title_out_microCDCN, output_microCDCN_path)



    print('---------------')
    microCDCN_channels = MicroCDCN_Channels(basic_conv=Conv2d_cd_channels, theta=0.7)
    out_microCDCN_channels = microCDCN_channels(im_rgb)
    out_microCDCN_channels = (out_microCDCN_channels*255).float()
    # print('out_microCDCN_channels:', out_microCDCN_channels)
    # for param in microCDCN_channels.parameters():
    #     print('microCDCN_channels:', param)
    #     print('microCDCN_channels:', param.shape)
    print('out_microCDCN_channels.shape:', out_microCDCN_channels.shape)
    title_out_microCDCN_channels = 'Central Difference Convolution Between Channels'
    plot_feature_maps_1_model(img, out_microCDCN_channels, title_out_microCDCN_channels, output_microCDCN_Channels_path)




    # title_out_microCDCN_channels = 'Central Difference Convolutions'
    # plot_feature_maps_2_model(img, out_microCDCN, out_microCDCN_channels, title_out_microCDCN_channels, output_2_models_path)
    
