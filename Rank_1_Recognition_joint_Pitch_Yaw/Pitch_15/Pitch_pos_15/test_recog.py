import os, sys, shutil
import struct as st
import numpy as np
from scipy import spatial
from sklearn import metrics
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from sklearn.metrics.pairwise import cosine_similarity


def load_meta_data(meta_file, sub_dir):
    meta_data = dict()
    with open(meta_file, 'r') as in_f:
        in_f.readline()  # the first line is not data
        for idx, line in enumerate(in_f):
            # record = line.strip().split(',')
            record = line.strip().split('/')[-1]
            template = int(record.split('_')[0])
            class_id = int(record.split('_')[0])
            # template, class_id = int(record[0]), int(record[1])
            img_path = '{}/{}'.format(sub_dir, record)
            if template not in meta_data:
                meta_data[template] = (class_id, [img_path, ])
            else:
                meta_data[template][1].append(img_path)
    return meta_data


def load_feat(list_file, bin_file):
    mid_feats = dict()
    with open(list_file, 'r') as list_f, open(bin_file, 'rb') as bin_f:
        (data_num, feat_dim) = st.unpack('ii', bin_f.read(8))
        for line in list_f:
            record = line.strip().split('/')
            img_name = '/'.join(record[-2:])
            feat = np.array(st.unpack('f' * feat_dim, bin_f.read(4 * feat_dim)))
            mid_feats[img_name] = feat
    return mid_feats, feat_dim


def update_meta_data(meta_data, feats, feat_dim):
    new_meta_data = dict()
    for template in meta_data.keys():
        class_id, img_names = meta_data[template]
        feat = np.zeros(feat_dim)
        feat_num = 0
        for img_name in img_names:
            if img_name in feats:
                feat += feats[img_name]
                feat_num += 1
        if feat_num > 0: feat /= feat_num
        if feat_num > 0: new_meta_data[template] = (class_id, feat)
    return new_meta_data


def get_top(probe, gallery_data):
    score_info = list()
    probe_id, probe_feat = probe
    for template in gallery_data.keys():
        gallery_id, gallery_feat = gallery_data[template]
        score = 1 - spatial.distance.cosine(probe_feat, gallery_feat)
        score_info.append((gallery_id, score))
    score_info = sorted(score_info, key=lambda a: a[1], reverse=True)
    top5_id = [item[0] for item in score_info[:5]]
    return top5_id


def eval_recog(probe_data, gallery_data):
    gallery_ids = set()

    for template in gallery_data.keys():
        gallery_ids.add(gallery_data[template][0])
    top1_num, top5_num, tot_num = 0, 0, 0
    for template in probe_data.keys():
        class_id = probe_data[template][0]
        if class_id not in gallery_ids: continue
        top5_id = get_top(probe_data[template], gallery_data)
        if class_id == top5_id[0]:
            top1_num += 1
            top5_num += 1
        elif class_id in top5_id:
            top5_num += 1
        tot_num += 1
    return top1_num / tot_num, top5_num / tot_num


def test_recog(model_type):
    # model_type = 'resnet18'
    # protocol_dir = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/IJBA/IJBA/protocol_1N'
    # align_img_dir = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/IJBA/IJBA/align_image_1N'
    # split_num = 10
    top1s, top5s = list(), list()
    # for split in range(1, split_num + 1):

    # load meta data first
    # split_protocol_dir = os.path.join(protocol_dir, 'split{}'.format(split))
    # split_img_dir = os.path.join(align_img_dir, 'Split{}'.format(Split))
    # probe_file = os.path.join(split_protocol_dir, 'search_probe_{}.csv'.format(split))
    Rank_1_dir = "/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition_joint_Pitch_Yaw/Pitch_15/Pitch_pos_15"
    probe_file = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition_joint_Pitch_Yaw/Pitch_15/Pitch_pos_15/probe_60_list.txt'
    probe_data = load_meta_data(probe_file, 'probe_60')
    # gallery_file = os.path.join(split_protocol_dir, 'search_gallery_{}.csv'.format(split))
    gallery_file = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition_joint_Pitch_Yaw/Pitch_15/Pitch_pos_15/Gallery_list.txt'
    gallery_data = load_meta_data(gallery_file, 'Gallery')
    # load extract feat
    feats = dict()
    # feat_dim = 0
    # split_img_dir = os.path.join(align_img_dir, 'split{}'.format(split))
    for img_type in ['Gallery', 'probe_60']:
        list_file = os.path.join(Rank_1_dir, '{}_list.txt'.format(img_type))
        bin_file = os.path.join(Rank_1_dir, '{}_{}_feat.bin'.format(model_type, img_type))
        mid_feats, feat_dim = load_feat(list_file, bin_file)
        feats.update(mid_feats)

    # update meta data
    probe_data = update_meta_data(probe_data, feats, feat_dim)
    gallery_data = update_meta_data(gallery_data, feats, feat_dim)
    top1, top5 = eval_recog(probe_data, gallery_data)
    top1s.append(top1)
    top5s.append(top5)
    print(' top1: {}, top5: {}'.format(top1, top5))

    print('top1: {} +/- {}'.format(np.mean(top1s), np.std(top1s)))
    print('top5: {} +/- {}'.format(np.mean(top5s), np.std(top5s)))

    return np.mean(top1s), np.std(top1s), np.mean(top5s), np.std(top5s)


if __name__ == '__main__':
    test_recog(model_type='Face_Pose_AwareNet')
