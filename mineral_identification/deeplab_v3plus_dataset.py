import torch
from torch import nn
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image 
import random
import os
import cv2
import numpy as np
from osgeo import gdal
from layer_data_augmentator import DataAugmentation
class deeplabv3plus_dataset(Dataset.Dataset):
    # csv_dir1: PPL/XPL images,labels  csv_dir2: boundary images
    def __init__(self, csv_dir1,csv_dir2, gpu=True):
        self.csv_dir1 = csv_dir1
        self.csv_dir2 = csv_dir2          
        self.names_list1 = []
        self.names_list2 = []          
        self.size = 0
        self.gpu = gpu
        self.img_num = 0
        if not os.path.isfile(self.csv_dir1):
            print(self.csv_dir1 + ':txt file does not exist!')
        if not os.path.isfile(self.csv_dir2):
            print(self.csv_dir2 + ':txt file does not exist!')
        file1 = open(self.csv_dir1)
        file2 = open(self.csv_dir2)
        
        for f in file1:
            self.names_list1.append(f)
            self.size += 1
        for f in file2:
            self.names_list2.append(f)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path1 = self.names_list1[idx].split(',')[0]
        img_path2 = self.names_list2[idx].strip('\n')
        img_raw1 = gdal.Open(img_path1)
        img_raw2 = gdal.Open(img_path2)
        img_w = img_raw1.RasterXSize
        img_h = img_raw1.RasterYSize
        label_path = self.names_list1[idx].split(',')[1].strip('\n')
        label_raw = gdal.Open(label_path)

        sample1 = {'raw_image':[], 'img': [], 'label': []}
        sample2 = {'raw_image':[], 'img': []}
        img1 = np.array(img_raw1.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        img2 = np.array(img_raw2.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        label = np.array(label_raw.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('uint8')
        
        # Normalization
        img1 = (img1-np.min(img1))/(np.max(img1)-np.min(img1))
       
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2).unsqueeze(0)
        raw_image1 = img1
        raw_image2 = img2
        label = torch.from_numpy(label)
        label = label.contiguous().view(label.size()[0],label.size()[1])
        
        img1 = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(img1)
        
        # concatenate PPL/XPL images and boundary images
        img1 = torch.cat((img1,img2),0)

        if self.gpu == True:
            img1 = img1.cuda()
            img2 = img2.cuda()
            label = label.cuda()

        sample1['raw_image']=raw_image1
        sample1['img']=img1
        sample1['label']=label
        sample2['raw_image']=raw_image2
        sample2['img']=img2
        
        return sample1,sample2


def add_conv_channels(model, premodel, conv_num):
    model_dict = model.state_dict()
    premodel_dict = premodel.state_dict()

    for i in range(conv_num[0]):
        conv = torch.FloatTensor(64,1,3,3).cuda()
        nn.init.xavier_normal_(conv)

        orginal1 = premodel_dict['conv_1.0.weight']
        new = torch.cat([orginal1,conv],1)
        premodel_dict['conv_1.0.weight'] = new

    model.load_state_dict(premodel_dict)
    print('set model with predect model, add channel is ',conv_num)
    return model

def gamma_transform(img, gamma=0.8):
    img /= 255
    img = np.power(img, gamma)
    img *= 255
    return img

def compress_graylevel(img, input_graylevel, output_graylevel):
    print("---doing compress_graylevel---")
    rate = input_graylevel/output_graylevel
    img = img//rate
    img = img*rate
    return img

def main():
    dataset = deeplabv3plus_dataset('train_xpl.csv', 3)
    for i, data in enumerate(dataset):
        raw_images, images, labels = data['raw_image'], data['img'], data['label']
        # print(raw_images.shape, images.shape, labels.shape)
    # a = next(dataset)
    # print(a)
    # print(len(a))
if __name__ == '__main__':
    main()
