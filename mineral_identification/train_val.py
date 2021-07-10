from tqdm import tqdm
import network
import utils
import os
import random
import time
import argparse
import numpy as np

from torch.utils import data
import torchvision.transforms as transforms
import torch
import torch.nn as nn

from PIL import Image
from deeplab_v3plus_dataset import deeplabv3plus_dataset

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--num_classes", type=int, default=6,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--input_channels", type=int, default=6,
                        help="input_channels")
    parser.add_argument("--total_itrs", type=int, default=273750,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 2)')
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    return parser

def main():
    opts = get_argparser().parse_args()
    print(opts)
    print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

    train_dst = deeplabv3plus_dataset('train_ppl.csv','train_edge.csv')
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=False, drop_last=True)
    
    val_dst = deeplabv3plus_dataset('val_ppl.csv','val_edge.csv')
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, drop_last=True)
    
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model_img = model_map[opts.model](input_channels=4,num_classes=9, output_stride=opts.output_stride)
    
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model_img.classifier)
    utils.set_bn_momentum(model_img.backbone, momentum=0.01)

    # Set up optimizer
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model_img.backbone.parameters(), 'lr': 0.1*opts.lr},
    #     {'params': model_img.classifier.parameters(), 'lr': opts.lr}
    # ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # if opts.lr_policy=='poly':
    #     scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    # elif opts.lr_policy=='step':
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    optimizer = torch.optim.Adam(params=[
        {'params': model_img.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model_img.classifier.parameters(), 'lr': opts.lr}
    ], lr=opts.lr, weight_decay=opts.weight_decay)

    num_classes = 9
    pixelsum = torch.zeros(num_classes, dtype=torch.int64)
    for i, (sample1, sample2) in enumerate(train_loader, 0):
        labels = sample1['label']
        for classes in range(num_classes):
            class_index = labels == classes
            pixelsum[classes] += int(class_index.sum())
    print(pixelsum)
    loss_function = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    cur_itrs = 0
    cur_epochs = 0
    #==========   Train Loop   ==========#
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  ===== #
        model_img.train()
        cur_epochs += 1
        interval_loss = 0
        model_img = model_img.cuda()
        for i, (sample1, sample2) in tqdm(enumerate(train_loader, 0), desc='Training:', total=len(train_loader)):
            cur_itrs += 1
            optimizer.zero_grad()

            raw_images1, images1, labels = sample1['raw_image'], sample1['img'], sample1['label']
            raw_images2, images2 = sample2['raw_image'], sample2['img']
            outputs = model_img(images1,images2)
            loss = loss_function(outputs, labels.long())
            loss.backward()
            optimizer.step()

            loss = loss.detach().cpu().numpy()
            interval_loss += loss

        print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, loss is %f'%(cur_epochs, interval_loss))

        #=======  Validation   ======#
        if cur_epochs % 5 == 0:
            model_img.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for j, (val_sample1, val_sample2) in enumerate(val_loader, 0):
                    val_images1, val_labels = val_sample1['img'], val_sample1['label']
                    val_images2 = val_sample2['img']
                    val_outputs = model_img(val_images1,val_images2)
                    val_loss = loss_function(val_outputs, val_labels.long())
                    val_loss = val_loss.detach().cpu().numpy()
                    val_loss_sum += val_loss
            
            print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()), 'epoch is %d, val_loss is %f'%(cur_epochs, val_loss_sum))
            image_model_name = 'result\\deeplabv3_plus_image_model' + str(cur_epochs) + '.pkl'
            torch.save(model_img, image_model_name)
        
        if cur_epochs >= 0:
            img_fn1 = 'result\\' + str(cur_epochs) + 'img.png'
            img_fn2 = 'result\\' + str(cur_epochs) + 'edge.png'
            lab_fn = 'result\\' + str(cur_epochs) + 'lab.png'
            
            out_img1 = raw_images1.cpu()
            out_img1 = transforms.ToPILImage()(out_img1[0])
            out_img1.save(img_fn1)
            out_img2 = raw_images2.cpu()
            out_img2 = transforms.ToPILImage()(out_img2[0])
            out_img2.save(img_fn2)

            out_lab = labels.cpu()
            out_lab = transforms.ToPILImage()(out_lab[0])
            out_lab.save(lab_fn)

            # flag = 0
            for item in [outputs]:
                fn = 'result\\' + str(cur_epochs) + '.png'
                predects = item[0]
                _, predects = torch.max(item[0], 0)
                predects = predects.int()
                predects = predects.cpu()
                predects = transforms.ToPILImage()(predects)
                predects.save(fn)
                # flag += 1
        
        if cur_itrs >=  opts.total_itrs:
            return

if __name__ == '__main__':
    main()
