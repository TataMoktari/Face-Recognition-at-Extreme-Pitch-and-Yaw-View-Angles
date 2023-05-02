import argparse
# from utils import *
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Facenet_tune import FacePoseAwareNet, PFDiscriminator
import torch.backends.cudnn as cudnn

from contrastive_with_memory import ContrastMemory
from dset import CMUPIE_SupConMemory

from utils import *
from contrastive_dataset_generation import get_dataset
from train_dataset import get_dataset
from validation_dataset import get_dataset_val
from torchvision import transforms
import os
from torch import optim
import torch.backends.cudnn as cudnn

#################################################################################################
parser = argparse.ArgumentParser(description='Contrastive view')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--frontal_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/Contrastive_Dataset/cfp-dataset/frontal_test_cropped',
                    help='path to data')
parser.add_argument('--profile_folder', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/Contrastive_Dataset/cfp-dataset/profile_test_cropped',
                    help='path to data')
parser.add_argument('--frontal_dir', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/m2fpa_frontal_train.csv',
                    help='path to data')
parser.add_argument('--profile_dir', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/m2fpa_profile_train.csv',
                    help='path to data')
parser.add_argument('--lambda_ADV', type=float, default=0.1, help='ADV loss * lambda_ADV')
parser.add_argument('--lambda_SCLM', type=float, default=1.0, help='SupConMemory loss coefficient')
parser.add_argument('--lambda_SCL', type=float, default=1.0, help='SupCon loss coefficient')
parser.add_argument('--temperature', type=float, default=7e-2, help='temperature in SCL')
parser.add_argument('--nce_k', type=int, default=200, help='number of negative samples')
parser.add_argument('--num_positive', type=int, default=1, help='number of positive samples')
parser.add_argument('--embedding_size', type=int, default=512, help='embedding size of the output')
parser.add_argument('--resume', type=str, required=False, help='resume training?')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--lr_d', type=float, default=1e-3, help='initial learning rate for discriminator')
parser.add_argument('--class_num', type=int, default=7883, help='number of classes in the dataset')
parser.add_argument('--save_dir', type=str,
                    default='/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/logfile/',
                    help='path to save the data')

args = parser.parse_args()
############################################################
# SET UP Pose Attention-Guided Deep Subspace Learning for PIFR #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
resnet = FacePoseAwareNet(pose=None)
if torch.cuda.device_count() > 1:  ##  to use both GPUs if available
    print("CHECKING GPUS  AVAILABLE")
    print(torch.cuda.device_count())
    resnet = nn.DataParallel(resnet, device_ids=list(range(torch.cuda.device_count())))
resnet = resnet.to(device)
cudnn.benchmark = True
resnet.train()


####################DataLoader-Initialization#################

def fixed_image_standardization(image_tensor):
    # processed_tensor = (image_tensor - 127.5) / 128.0
    processed_tensor = (image_tensor - .5) / .5
    return processed_tensor


crop_size = (112, 112)
trf_train = transforms.Compose(
    [transforms.Resize((120, 120)), transforms.CenterCrop(crop_size),
     transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
frontal_train = pd.read_csv(
    '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/m2fpa_frontal_train.csv')
profile_train = pd.read_csv(
    '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/m2fpa_profile_train.csv')
train_dset = CMUPIE_SupConMemory(profile_df=profile_train, frontal_df=frontal_train,
                                 transform=trf_train,
                                 nce_k=args.nce_k, num_positive=args.num_positive,
                                 num_classes=args.class_num)
train_loader = DataLoader(dataset=train_dset,
                          batch_size=args.bs,
                          shuffle=True)

#################*****Memory Bank*******##################
memory_bank_loss = ContrastMemory(inputSize=args.embedding_size,
                                  K=args.nce_k,
                                  T=args.temperature,
                                  momentum=0.5,
                                  base_temperature=0.07,
                                  frontal_csv=args.frontal_dir,
                                  profile_csv=args.profile_dir,
                                  resume=args.resume,
                                  device=device,
                                  num_positive=args.num_positive).cuda()

####################Hyperparameters-Initialization############
argmargin = 1.4
lr = 0.0001
gamma = 0.01
epochs = 15
patience = 15

##########check parameters required gradient#############
# for name, param in resnet.module.named_parameters():
for name, param in resnet.named_parameters():
    if param.requires_grad:
        print(name)

optimizer = optim.Adam(resnet.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

import matplotlib.pyplot as plt


def norm_minmax(x):
    """
    min-max normalization of numpy array
    """
    return (x - x.min()) / (x.max() - x.min())


def plot_tensor(t):
    """
    plot pytorch tensors
    input: list of tensors t
    """
    for i in range(len(t)):
        ti_np = t[i].cpu().detach().numpy().squeeze()
        ti_np = norm_minmax(ti_np)
        if len(ti_np.shape) > 2:
            ti_np = ti_np.transpose(1, 2, 0)
        plt.subplot(1, len(t), i + 1)
        plt.imshow(ti_np)
    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


###############################################################

val_loader = get_dataset_val(args)


def validate(epoch):
    resnet.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    for iter, (img_photo, img_morph, lbl) in enumerate(val_loader):
        bs = img_photo.size(0)
        lbl = lbl.type(torch.float)
        img_photo, img_morph, lbl = img_photo.to(device), img_morph.to(device), lbl.to(device)
        y_photo = resnet(img_photo, pose='frontal')
        y_morph = resnet(img_morph, pose='profile')
        dist = ((y_photo - y_morph) ** 2).sum(1)
        margin = torch.ones_like(dist, device=device) * argmargin
        loss = lbl * dist + (1 - lbl) * F.relu(margin - dist)
        loss = loss.mean()
        acc = (dist < argmargin).type(torch.float)
        acc = (acc == lbl).type(torch.float)
        acc = acc.mean()
        acc_m.update(acc)
        loss_m.update(loss.item())
    print('VALIDATION epoch: %02d, loss: %.4f, acc: %.4f' % (epoch, loss_m.avg, acc_m.avg))
    return loss_m.avg, acc_m.avg


##########################*******Discriminator*******#####################################
netCritic = PFDiscriminator(num_input=args.embedding_size).cuda()
# optimizer_d = optim.SGD(netCritic.parameters(), lr=args.lr_d, momentum=0.9, weight_decay=1e-5)
optimizer_d = optim.Adam(netCritic.parameters(), lr=args.lr_d, weight_decay=1e-5)
###########################################################################
println = len(train_loader) // 5
print(println)
chkloss = 100
step = 0
pl = 0
best_acc = 0
best_epoch = 0
best_all = []
all_step = 0

log_name = os.path.join(args.save_dir, 'loss_log_train.txt')

tensorboard_dir = os.path.join(args.save_dir, 'tboard')
if not os.path.exists(tensorboard_dir):
    os.mkdir(tensorboard_dir)
writer = SummaryWriter(tensorboard_dir)


def log_loss_tensorboard(self, epoch, writer):
    loss_dict = {'critic_profile': self.epoch_critic_profile.avg,
                 'adv_profile': self.epoch_adv_profile.avg,
                 'SupConM': self.epoch_SCLM.avg}
    for k, v in loss_dict.items():
        writer.add_scalars(k, {k: v}, epoch)


# ce_loss = nn.CrossEntropyLoss()
ce_loss = nn.BCELoss()


def reset_losses(self):
    self.epoch_ce = AverageMeter()
    self.epoch_SCLM = AverageMeter()
    self.epoch_critic_profile = AverageMeter()
    self.epoch_critic_frontal = AverageMeter()
    self.epoch_adv_profile = AverageMeter()


for epoch in range(epochs):
    print('Ready to train......')
    resnet.train()
    netCritic.train()
    epoch_adv_profile = AverageMeter()
    epoch_ce = AverageMeter()
    epoch_SCLM = AverageMeter()
    epoch_critic_profile = AverageMeter()
    epoch_critic_frontal = AverageMeter()
    print('iteration starts...')

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        profile_face = data['profile'].to(device)
        frontal_face = data['frontal'].to(device)
        lbl = data['lbl'].to(device)
        idp = data['id'].to(device)
        idf = idp
        neg_frontal = data['neg_idx_frontal'].to(device)
        neg_profile = data['neg_idx_profile'].to(device)
        y1 = data['y1'].to(device)
        y2 = data['y2'].to(device)
        frontal_embeddings = resnet(frontal_face, pose='frontal')
        profile_embeddings = resnet(profile_face, pose='profile')
        label_frontal = torch.ones(frontal_embeddings.size(0)).float()
        label_frontal = Variable(label_frontal.to(device))
        label_frontal = label_frontal.unsqueeze(1)
        pred_profile = netCritic(profile_embeddings)
        loss_adv_profile = ce_loss(pred_profile,
                                   label_frontal) * args.lambda_ADV  # critic should not discriminate between profile and frontal images
        epoch_adv_profile.update(loss_adv_profile.item(), profile_face.shape[0])
        SCL_memory = memory_bank_loss(v1=frontal_embeddings, y1=y1,
                                      v2=profile_embeddings, y2=y2,
                                      idx1=neg_frontal, idx2=neg_profile,
                                      opt=device) * args.lambda_SCLM

        epoch_SCLM.update(SCL_memory.item(), profile_face.shape[0])
        loss_total = SCL_memory + loss_adv_profile
        loss_total.backward()
        optimizer.step()

        #######################Discriminator Backward###############
        optimizer_d.zero_grad()
        label_profile = torch.zeros(profile_embeddings.size(0)).float()
        label_profile = Variable(label_profile.to(device))
        label_profile = label_profile.unsqueeze(1)
        pred_frontal = netCritic(frontal_embeddings.detach())
        pred_profile = netCritic(profile_embeddings.detach())
        loss_critic_frontal = ce_loss(pred_frontal, label_frontal)
        loss_critic_profile = ce_loss(pred_profile, label_profile)
        epoch_critic_profile.update(loss_critic_profile.item(), profile_face.shape[0])
        epoch_critic_frontal.update(loss_critic_frontal.item(), frontal_face.shape[0])
        loss_critic = (loss_critic_frontal + loss_critic_profile) / 2
        loss_critic.backward()
        optimizer_d.step()
        ##################**************#################################
        if iter % println == 0:
            loss_dict = {'critic_profile': epoch_critic_profile.avg,
                         'critic_frontal': epoch_critic_frontal.avg,
                         'adv_profile': epoch_adv_profile.avg,
                         'SupConM': epoch_SCLM.avg}
            # loss_dict = {'SupConM': epoch_SCLM.avg}
            message = '(epoch: %d, iter: %d/%d, ) ' % (epoch, iter, len(train_loader))
            for k, v in loss_dict.items():
                message += '%s: %.3f ' % (k, v)
            print(message)  # print the message
            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % message)

    state = {}
    state['resnet'] = resnet.state_dict()
    state['optimizer'] = optimizer.state_dict()
    val_loss, val_acc = validate(epoch)
    if val_loss > chkloss:
        print("STEP " + str(step + 1) + "\tPLATEAU: " + str(pl) + "\tLR: " + str(lr))
        step += 1
        all_step += 1
        if step > patience:
            best_all.append([best_epoch, chkloss, best_acc, best_weights])
            scheduler.step()
            print("PLATEAU: LOWERING LR...")
            lr = lr * gamma
            #### CONTINUE TRAINING ON LOWER LR FROM THE BEST SAVED WEIGHTS ###
            resnet.load_state_dict(torch.load(best_weights)['resnet'])
            step = 0
            pl += 1
            if all_step > (patience * 2):
                break

    else:
        chkloss = val_loss
        step = 0
        all_step = 0
        best_acc = val_acc
        best_epoch = epoch
        best_weights = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/checkpoint/' + str(
            args.batch_size) + '_LR' + str(lr) + str(
            epoch) + '_VALID_BEST.pt'
        torch.save(state, best_weights)
    print('\n Model Saved! \n')

FINAL_WEIGHTS = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_attention_7x7_Memory/checkpoint/' + str(
    args.batch_size) + '_LR' + str(lr) + '_FINAL_WEIGHTS_VALID' + '.pth'
torch.save(resnet.state_dict(), FINAL_WEIGHTS)
best_all.append([best_epoch, chkloss, best_acc, best_weights])

for best in best_all:
    print(best)
