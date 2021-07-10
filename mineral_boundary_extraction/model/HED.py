import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HED(nn.Module):
    def __init__(self, input_channels):
        super(HED,self).__init__()
        self.input_channels = input_channels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv_4 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
        )
        self.conv_5 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


        self.deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)


        self.sideout_conv1 = nn.Conv2d(64, 1, 1)
        self.sideout_conv2 = nn.Conv2d(128, 1, 1)
        self.sideout_conv3 = nn.Conv2d(256, 1, 1)
        self.sideout_conv4 = nn.Conv2d(512, 1, 1)
        self.sideout_conv5 = nn.Conv2d(512, 1, 1)
        self.sideout_fuse = nn.Conv2d(5, 1, 1)


        # self.Combine = torch.nn.Sequential(
		# 	nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
		# 	nn.Sigmoid()
		# )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0, 0.2)
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
        
        if self.input_channels == 3:
            HED_load_premodel(self,'C:\\Users\\ASUS\\Desktop\\mineral_boundary_extraction\\VGG_ILSVRC_16_layers.npy')

    def forward(self, x):

        img_h = x.size(2)
        img_w = x.size(3)

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.conv_3(x2)
        x4 = self.conv_4(x3)
        x5 = self.conv_5(x4)

        sideout_1 = self.sideout_conv1(x1)
        sideout_2_upsample = F.interpolate(self.sideout_conv2(x2), size=(img_h, img_w), mode='bilinear')
        sideout_3_upsample = F.interpolate(self.sideout_conv3(x3), size=(img_h, img_w), mode='bilinear')
        sideout_4_upsample = F.interpolate(self.sideout_conv4(x4), size=(img_h, img_w), mode='bilinear')
        sideout_5_upsample = F.interpolate(self.sideout_conv5(x5), size=(img_h, img_w), mode='bilinear')

        # sideout_1 = self.sideout_conv1(x1)
        # sideout_2_upsample = self.deconv(self.sideout_conv2(x2))
        # sideout_3_upsample = self.deconv(self.deconv(self.sideout_conv3(x3)))
        # sideout_4_upsample = self.deconv(self.deconv(self.deconv(self.sideout_conv4(x4))))
        # sideout_5_upsample = self.deconv(self.deconv(self.deconv(self.deconv(self.sideout_conv5(x5)))))


        sideout_concat = self.sideout_fuse(torch.cat((sideout_1,sideout_2_upsample,sideout_3_upsample,sideout_4_upsample,sideout_5_upsample), 1))

        # sideout1 = torch.sigmoid(sideout_1)
        # sideout2 = torch.sigmoid(sideout_2_upsample)
        # sideout3 = torch.sigmoid(sideout_3_upsample)
        # sideout4 = torch.sigmoid(sideout_4_upsample)
        # sideout5 = torch.sigmoid(sideout_5_upsample)
        # sideoutcat = torch.sigmoid(sideout_concat)

        sideout1 = sideout_1
        sideout2 = sideout_2_upsample
        sideout3 = sideout_3_upsample
        sideout4 = sideout_4_upsample
        sideout5 = sideout_5_upsample
        sideoutcat = sideout_concat

        return sideout1, sideout2, sideout3, sideout4, sideout5, sideoutcat

class HED_fuse(nn.Module):
    def __init__(self, input_channels1,input_channels2):
        super(HED_fuse,self).__init__()
        self.HED1=HED(input_channels1)
        self.HED2=HED(input_channels2)
        self.layer1 = nn.Conv2d(2,1,1)
        self.layer2 = nn.Conv2d(2,1,1)
        self.layer3 = nn.Conv2d(2,1,1)
        self.layer4 = nn.Conv2d(2,1,1)
        self.layer5 = nn.Conv2d(2,1,1)
        self.layer6 = nn.Conv2d(2,1,1)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm2d(1)
        self.bn4 = nn.BatchNorm2d(1)
        self.bn5 = nn.BatchNorm2d(1)
        self.bn6 = nn.BatchNorm2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                # m.weight.data.normal_(0, 0.2)
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x1, x2):
        sideout1_1, sideout1_2, sideout1_3, sideout1_4, sideout1_5, sideoutcat1 = self.HED1(x1)
        sideout2_1, sideout2_2, sideout2_3, sideout2_4, sideout2_5, sideoutcat2 = self.HED2(x2)
        sideout_fuse_1 = torch.cat((sideout1_1, sideout2_1), dim=1)
        sideout_fuse_2 = torch.cat((sideout1_2, sideout2_2), dim=1)
        sideout_fuse_3 = torch.cat((sideout1_3, sideout2_3), dim=1)
        sideout_fuse_4 = torch.cat((sideout1_4, sideout2_4), dim=1)
        sideout_fuse_5 = torch.cat((sideout1_5, sideout2_5), dim=1)
        sideout_fuse_6 = torch.cat((sideoutcat1, sideoutcat2), dim=1)
        sideout_fuse_1 = self.layer1(sideout_fuse_1)
        sideout_fuse_2 = self.layer2(sideout_fuse_2)
        sideout_fuse_3 = self.layer3(sideout_fuse_3)
        sideout_fuse_4 = self.layer4(sideout_fuse_4)
        sideout_fuse_5 = self.layer5(sideout_fuse_5)
        sideout_fuse_6 = self.layer6(sideout_fuse_6)
        sideout_fuse_1 = self.bn1(sideout_fuse_1)
        sideout_fuse_2 = self.bn2(sideout_fuse_2)
        sideout_fuse_3 = self.bn3(sideout_fuse_3)
        sideout_fuse_4 = self.bn4(sideout_fuse_4)
        sideout_fuse_5 = self.bn5(sideout_fuse_5)
        sideout_fuse_6 = self.bn6(sideout_fuse_6)
        sideout_fuse_1 = torch.sigmoid(sideout_fuse_1)
        sideout_fuse_2 = torch.sigmoid(sideout_fuse_2)
        sideout_fuse_3 = torch.sigmoid(sideout_fuse_3)
        sideout_fuse_4 = torch.sigmoid(sideout_fuse_4)
        sideout_fuse_5 = torch.sigmoid(sideout_fuse_5)
        sideout_fuse_6 = torch.sigmoid(sideout_fuse_6)

        return sideout_fuse_1, sideout_fuse_2, sideout_fuse_3, sideout_fuse_4, sideout_fuse_5, sideout_fuse_6


def HED_LOSS(input, target):
    n, c, h, w = input.size()
    target = target.reshape([n, c, h, w]).float()
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_trans = target_t.clone()
    pos_index = (target_t > 0)
    neg_index = (target_t == 0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num*1.0 / sum_num
    weight[neg_index] = pos_num*1.0 / sum_num
    
    weight = torch.from_numpy(weight)
    weight = weight.cuda()
    loss = F.binary_cross_entropy(log_p, target_t, weight, size_average=True)
    return loss


def HED_load_premodel(model, premodel_filename):
    new_params = np.load(premodel_filename, allow_pickle=True, encoding='bytes')
    model_dict = model.state_dict()
    premodel_dict = new_params[0]
    premodel_list = []
    for key, value in premodel_dict.items():
        temp_dict = {'key':key,'value':value}
        premodel_list.append(temp_dict)
    param_layer = 0
    for key in model_dict:
        if 'deconv' in key:
            break
        if 'conv' in key:
            pre_k = premodel_list[param_layer]['key']
            pre_v = premodel_list[param_layer]['value']
            pre_v = torch.from_numpy(pre_v)
            assert model_dict[key].shape == pre_v.shape
            model_dict[key] = pre_v
            #print('     set FCN model %s layer param by Pascal premodel %s layer param'%(key, pre_k))
            param_layer += 1
        
    model.load_state_dict(model_dict)
    print('HED init weight by %s model'%premodel_filename)
    return model

def main():
    model = HED_fuse(3, 3)

if __name__ == '__main__':
    main()


