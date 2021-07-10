import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image 
import torchvision.models as models
from torchvision.models.vgg import VGG
import os
from osgeo import gdal
import numpy as np
import model.HED as HED
from tqdm import tqdm
import cv2
from osgeo import gdal

folder_path = 'C:\\Users\\ASUS\\Desktop\\boundary_extraction\\'
# model_path='E:\\HED_train_fuse\\'
# test_ppl_folder_name='test_ppl.csv'
# test_xpl_folder_name='test_xpl.csv'
# test_result_folder_name='test_result_HEDfuse'

class HED_dataset(Dataset.Dataset):
    def __init__(self, csv_dir1,csv_dir2, transform=None):
        self.csv_dir1 = folder_path + csv_dir1 
        self.csv_dir2 = folder_path + csv_dir2        
        self.names_list1 = []
        self.names_list2 = []
        self.size = 0
        self.transform = transform
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
        image_path1 = self.names_list1[idx].split(',')[0]
        image_path2 = self.names_list2[idx].split(',')[0]
        image1 = gdal.Open(image_path1)
        image2 = gdal.Open(image_path2)
        img_w = image1.RasterXSize
        img_h = image1.RasterYSize
        
        label_path = self.names_list1[idx].split(',')[1].strip('\n')
        label = Image.open(label_path)
        img1 = np.array(image1.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        img2 = np.array(image2.ReadAsArray(0,0,img_w,img_h,buf_xsize=img_w,buf_ysize=img_h)).astype('float32')
        # img1 = img1.transpose(1,2,0)
        # img2 = img2.transpose(1,2,0)
        image1 = torch.from_numpy(img1/255)
        image2 = torch.from_numpy(img2/255)
        sample1 = {'image': image1, 'label': label}
        sample2 = {'image': image2, 'label': label}
        sample1['image'] = self.transform(sample1['image'])
        sample2['image'] = self.transform(sample2['image'])
        sample1['label'] = torch.from_numpy(np.array(sample1['label']))
        sample2['label'] = sample1['label']
        return sample1,sample2

# def load_pretrained_model_vgg(model, pre_vggfile):
#     vgg_weight_layer_list = [0,3,7,10,14,17,20,24,27,30,34,37,40]
#     pretrained_VGG = torch.load(pre_vggfile)
#     layer_num = 0
#     model_dict = model.state_dict()
#     for key, value in model_dict.items():
#         if 'weight' in key and layer_num < len(vgg_weight_layer_list):
#             vgg_weight_dic_name = 'features.'+str(vgg_weight_layer_list[layer_num])+'.weight'
#             vgg_weight = pretrained_VGG[vgg_weight_dic_name]
#             assert value.size() == vgg_weight.size()
#             md1 = model_dict[key]
#             model_dict[key] = vgg_weight.cuda()
#             md2 = model_dict[key]
#             if md1.equal(md2):
#                 print('Assignment weight error')
#             layer_num += 1
#     model.load_state_dict(model_dict)
#     return model

Use_gpu = torch.cuda.is_available()

#train
if Use_gpu:
    Hed_IMG = HED.HED_fuse(input_channels1=3,input_channels2=3).cuda()
else:
    Hed_IMG = HED.HED_fuse(input_channels1=3,input_channels2=3)

#load_pretrained_model_vgg(Hed_IMG, 'pretrained\\vgg16.pth')
train_transforms = transforms.Compose([ transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
train_dataset = HED_dataset('train_ppl.csv','train_xpl.csv',transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# for i, sample in enumerate(train_dataloader, 0):
#     print(sample)
#     break

learn_rate = 0.001
optimizer = torch.optim.AdamW([{'params': Hed_IMG.parameters()}], lr=learn_rate, weight_decay=0)
count = 0
for epoch in range(500):
    epoch_loss_list = [0,0,0,0,0,0,0]
    for i, (sample1, sample2) in tqdm(enumerate(train_dataloader, 0), desc='Training:', total=len(train_dataloader)):
        optimizer.zero_grad()
        images1,images2,labels = sample1['image'],sample2['image'],sample1['label']
        
        if Use_gpu: 
            images1, images2, labels = images1.cuda(), images2.cuda(), labels.cuda()
        
        img_outputs = Hed_IMG(images1,images2)

        # a = img_outputs[4][0].cpu().detach().numpy() - img_outputs[5][0].cpu().detach().numpy()
        # a = a.transpose(1, 2, 0)
        # cv2.imshow('a', a)
        # cv2.waitKey(500)

        loss_side1 = HED.HED_LOSS(img_outputs[0], labels)
        loss_side2 = HED.HED_LOSS(img_outputs[1], labels)
        loss_side3 = HED.HED_LOSS(img_outputs[2], labels)
        loss_side4 = HED.HED_LOSS(img_outputs[3], labels)
        loss_side5 = HED.HED_LOSS(img_outputs[4], labels)
        final_loss = HED.HED_LOSS(img_outputs[5], labels)

        # add_outputall = img_outputs[0] + img_outputs[1] + img_outputs[2] + img_outputs[3] + img_outputs[4] + img_outputs[5]
        loss = (loss_side1 + loss_side2 + loss_side3 + loss_side4 + loss_side5 + final_loss)

        epoch_loss_list[0] += loss_side1.item()
        epoch_loss_list[1] += loss_side2.item()
        epoch_loss_list[2] += loss_side3.item()
        epoch_loss_list[3] += loss_side4.item()
        epoch_loss_list[4] += loss_side5.item()
        epoch_loss_list[5] += final_loss.item()
        epoch_loss_list[6] = epoch_loss_list[0]+epoch_loss_list[1]+epoch_loss_list[2]+epoch_loss_list[3]+epoch_loss_list[4]+epoch_loss_list[5]

        loss.backward()
        optimizer.step()

    
    print('epoch is %d : loss1 is %f , loss2 is %f , loss3 is %f , loss4 is %f , loss5 is %f , lossfuse is %f , lossall is %f'
            %(epoch,epoch_loss_list[0],epoch_loss_list[1],epoch_loss_list[2],epoch_loss_list[3],epoch_loss_list[4],epoch_loss_list[5],epoch_loss_list[6]))
    epoch_loss_list = [0,0,0,0,0,0,0]

    # if epoch % 100 == 0:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = learn_rate * 0.8

    if epoch % 1 == 0:
        layer = 0

        for i in [img_outputs[0], img_outputs[1], img_outputs[2], img_outputs[3], img_outputs[4], img_outputs[5]]:
            predicts = i.cpu().detach().numpy()*255
            predict = predicts[0].reshape(predicts[0].shape[1],predicts[0].shape[2])

            # bi = predict

            predict = transforms.ToPILImage()(predict)
            predict = predict.convert('RGB')
            predict_fn = folder_path + 'result\\'+str(count)+'_'+str(layer)+'pre.png'
            predict.save(predict_fn)
            
            layer += 1

        labels = labels.cpu().numpy() * 255
        label = labels[0].reshape(predicts[0].shape[1],predicts[0].shape[2])
        label = transforms.ToPILImage()(label)
        label = label.convert('RGB')
        label_fn =  folder_path + 'result\\'+str(count)+'lab.png'
        label.save(label_fn)
        
        img1 = transforms.ToPILImage()(images1[0].cpu())
        img_fn1 =  folder_path + 'result\\'+str(count) + 'pplimg.png'
        img1.save(img_fn1)

        img2 = transforms.ToPILImage()(images2[0].cpu())
        img_fn2 =  folder_path + 'result\\'+str(count) + 'xplimg.png'
        img2.save(img_fn2)

    count += 1

    if epoch % 50 == 0:
        image_model_name =  folder_path + 'Hed_image_model' + str(epoch) + '.pkl'
        torch.save(Hed_IMG, image_model_name)



# #test
# Hed_net = torch.load(model_path+'Hed_image_model350'+'.pkl').cuda()
# Hed_net.eval()

# test_transforms = transforms.Compose([transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
# test_dataset = HED_dataset(test_ppl_folder_name,test_xpl_folder_name,transform=test_transforms)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# if not os.path.exists(folder_path + test_result_folder_name):
#     os.mkdir(folder_path + test_result_folder_name)

# count = 0
# for i, (sample1, sample2) in enumerate(test_dataloader, 0):
#     images1,images2,labels = sample1['image'],sample2['image'],sample1['label']
#     images1 = images1.cuda()
#     images2 = images2.cuda()
#     # labels = labels.cuda()
#     outputs = Hed_net(images1,images2)
#     output_name = test_dataset.names_list1[i].split(',')[1].strip('\n')
#     output_name = output_name.split('\\')[-1].split('.')[0]
#     output_name = output_name.replace('SEM','EDGE')
     
#     final_output = outputs[5]
    
#     #_, predicts = torch.max(outputs, 1)

#     for j in [final_output]:
#         fuses_gray=j.cpu().detach().numpy()
#         imgw=fuses_gray[0].shape[1]
#         imgh=fuses_gray[0].shape[2]
#         fuse_gray = fuses_gray[0].reshape(imgw,imgh)

#         # create files
#         datatype=gdal.GDT_Float32
#         band_num=1
#         driver = gdal.GetDriverByName("GTiff")
#         fuse_gray_path=folder_path + test_result_folder_name+'\\' + output_name +'.tif'
#         dataset = driver.Create(fuse_gray_path, imgw, imgh, band_num, datatype)
#         if band_num==1:
#             dataset.GetRasterBand(1).WriteArray(fuse_gray)
#         else:
#             for i in range(band_num):
#                 dataset.GetRasterBand(i + 1).WriteArray(fuse_gray[i])
#         del dataset
    
#     count+=1

