import argparse
import os, sys, shutil
import time
import struct as st

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data

# from selfDefine import CFPDataset, CaffeCrop
# from ResNet import resnet18, resnet50, resnet101

from test_recog import test_recog
from Facenet_tune import FacePoseAwareNet

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
parser = argparse.ArgumentParser(description='Contrastive view')
parser.add_argument('--img_dir', metavar='DIR', default='', help='path to dataset')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--margin', default=0.85, type=int, help='batch size')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()
device = torch.device('cuda:0')

state = torch.load(
    '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/checkpoint/32_LR0.00016_VALID_BEST.pt')
resnet = FacePoseAwareNet(pose=None)
resnet = torch.nn.DataParallel(resnet)
resnet.to(device)
resnet.load_state_dict(state['resnet'])
resnet.eval()

print('load trained model complete')


def load_imgs(img_dir, image_list_file):
    imgs = list()
    with open(image_list_file, 'r') as imf:
        for line in imf:
            record = line.strip().split()
            # img_path, yaw = os.path.join(img_dir,record[0]), float(record[1])
            img_path = os.path.join(img_dir, record[0])
            imgs.append(img_path)
    return imgs


class CFPDataset(data.Dataset):
    def __init__(self, img_dir, image_list_file, transform=None):
        self.imgs = load_imgs(img_dir, image_list_file)
        self.transform = transform

    def __getitem__(self, index):
        # path, yaw = self.imgs[index]
        path = self.imgs[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


def fixed_image_standardization(image_tensor):
    # processed_tensor = (image_tensor - 127.5) / 128.0
    processed_tensor = (image_tensor - .5) / .5
    return processed_tensor


arch = 'Face_Pose_AwareNet'
root_dir = "/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition_joint_Pitch_Yaw/Pitch_30"
sub_dir = "Pitch_neg_30"
img_type_g = 'Gallery'
img_type_p = 'probe_45'

crop_size = 112, 112
split_dir = os.path.join(root_dir, sub_dir)
img_dir_g = os.path.join(split_dir, img_type_g)
img_dir_p = os.path.join(split_dir, img_type_p)
img_list_file_g = os.path.join(split_dir, '{}_list.txt'.format(img_type_g))
img_list_file_p = os.path.join(split_dir, '{}_list.txt'.format(img_type_p))
img_dataset_g = CFPDataset(args.img_dir, img_list_file_g,
                           transforms.Compose([transforms.Resize((120, 120)), transforms.CenterCrop(crop_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
img_dataset_p = CFPDataset(args.img_dir, img_list_file_p,
                           transforms.Compose([transforms.Resize((120, 120)), transforms.CenterCrop(crop_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
img_loader_g = torch.utils.data.DataLoader(
    img_dataset_g,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

img_loader_p = torch.utils.data.DataLoader(
    img_dataset_p,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

data_num = len(img_dataset_g)
img_feat_file_g = os.path.join(split_dir, '{}_{}_feat.bin'.format(arch, img_type_g))
feat_dim = 512
with open(img_feat_file_g, 'wb') as bin_f:
    bin_f.write(st.pack('ii', data_num, feat_dim))
    for i, input in enumerate(img_loader_g):
        input_var = torch.autograd.Variable(input, volatile=True)
        output = resnet(input_var, pose='frontal')
        output_data = output.cpu().data.numpy()
        feat_num = output.size(0)

        for j in range(feat_num):
            bin_f.write(st.pack('f' * feat_dim, *tuple(output_data[j, :])))

print('we have complete {}'.format(img_type_g))

data_num_p = len(img_dataset_p)
img_feat_file_p = os.path.join(split_dir, '{}_{}_feat.bin'.format(arch, img_type_p))
with open(img_feat_file_p, 'wb') as bin_f:
    bin_f.write(st.pack('ii', data_num_p, feat_dim))
    for i, input in enumerate(img_loader_p):
        input_var = torch.autograd.Variable(input, volatile=True)
        output = resnet(input_var, pose='profile')
        output_data = output.cpu().data.numpy()
        feat_num = output.size(0)

        for j in range(feat_num):
            bin_f.write(st.pack('f' * feat_dim, *tuple(output_data[j, :])))

print('we have complete {}'.format(img_type_p))

test_recog(arch)
