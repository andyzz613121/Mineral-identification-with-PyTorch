from tqdm import tqdm
import utils
import os
import numpy as np

from torch.utils import data
import torchvision.transforms as transforms

import torch
import torch.nn as nn

from PIL import Image
from deeplab_v3plus_dataset import deeplabv3plus_dataset

folder_path = 'test\\'
#==========   Test Loop   ==========#
model_img_ppl = torch.load('result\\deeplabv3_plus_image_model85.pkl').cuda()
model_img_xpl = torch.load('result\\deeplabv3_plus_image_model100.pkl').cuda()
model_img_ppl.eval()
model_img_xpl.eval()
test_dst_ppl = deeplabv3plus_dataset(folder_path+'test_ppl.csv',folder_path+'test_edge.csv')
test_dst_xpl = deeplabv3plus_dataset(folder_path+'test_xpl.csv',folder_path+'test_edge.csv')
test_loader_ppl = data.DataLoader(test_dst_ppl, batch_size=1, shuffle=False)
test_loader_xpl = data.DataLoader(test_dst_xpl, batch_size=1, shuffle=False)


with torch.no_grad():
    for i, (sample1_ppl,sample2_ppl) in tqdm(enumerate(test_loader_ppl, 0), desc='Testing:', total=len(test_loader_ppl)):
        raw_images1_ppl, images1_ppl, labels_ppl = sample1_ppl['raw_image'], sample1_ppl['img'], sample1_ppl['label']
        raw_images2_ppl, images2_ppl = sample2_ppl['raw_image'], sample2_ppl['img']
        outputs_ppl = model_img_ppl(images1_ppl,images2_ppl)

        sample1_xpl=test_dst_xpl[i]
        raw_images1_xpl, images1_xpl, labels_xpl = sample1_xpl[0]['raw_image'], sample1_xpl[0]['img'], sample1_xpl[0]['label']
        images1_xpl= images1_xpl.unsqueeze(0)
        outputs_xpl=model_img_xpl(images1_xpl,images2_ppl)

        lab_fn = folder_path + str(i) + 'lab.png'

        out_lab_ppl = labels_ppl.cpu()
        out_lab_ppl = transforms.ToPILImage()(out_lab_ppl[0])
        out_lab_ppl.save(lab_fn)
        
        predict_ppl = outputs_ppl[0]
        predict_xpl = outputs_xpl[0]

        predict_fuse=0.4*predict_ppl+0.6*predict_xpl
        _, predict_fuse=torch.max(predict_fuse,0)

        # # save ppl predicts
        # _, predict_ppl = torch.max(predict_ppl, 0)
        # fn_ppl = folder_path + str(i) + 'pre_ppl.png'
        # predict_ppl = predict_ppl.int()
        # predict_ppl = predict_ppl.cpu()
        # predict_ppl = transforms.ToPILImage()(predict_ppl)
        # predict_ppl.save(fn_ppl)

        # # save xpl predicts
        # _, predict_xpl = torch.max(predict_xpl, 0)
        # fn_xpl = folder_path + str(i) + 'pre_xpl.png'
        # predict_xpl = predict_xpl.int()
        # predict_xpl = predict_xpl.cpu()
        # predict_xpl = transforms.ToPILImage()(predict_xpl)
        # predict_xpl.save(fn_xpl)

        # save fused predicts
        fn_fuse = folder_path + 'fuse\\' + str(i) + 'pre_fuse.png'
        predict_fuse = predict_fuse.int()
        predict_fuse = predict_fuse.cpu()
        predict_fuse = transforms.ToPILImage()(predict_fuse)
        predict_fuse.save(fn_fuse)