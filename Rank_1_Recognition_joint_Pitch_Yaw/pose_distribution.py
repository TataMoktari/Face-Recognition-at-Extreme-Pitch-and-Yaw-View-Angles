import os
import shutil

source = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA_Test_cropped'

for file in sorted(os.listdir(source)):
    file_path = os.path.join(source, file)
    # file_w = os.path.join(file_path, 'White')
    # file_h = os.path.join(file_w, 'WearGlasses')
    for img in sorted(os.listdir(file_path)):
        img_s = img.split('.')[0]
        if '2-4' in img_s:
            img_path = os.path.join(file_path, img)
            dst_dir = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Pitch_30/probe_22.5/'
            new_name = file + '_' + img
            dst_path = os.path.join(dst_dir, new_name)
            shutil.copy(img_path, dst_path)
        elif '2-6' in img_s:
            img_path = os.path.join(file_path, img)
            dst_dir = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Pitch_30/probe_22.5/'
            new_name = file + '_' + img
            dst_path = os.path.join(dst_dir, new_name)
            shutil.copy(img_path, dst_path)

        # elif '4-7' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition/gallery/'
        #     new_name = file + '_' + img_s + '-w' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)

        # elif '5-7' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition/probe_15_pos/'
        #     new_name = file + '_' + img_s + '-w' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        #
        # elif '6-5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition/probe_30_pos/'
        #     new_name = file + '_' + img_s + '-w' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)

        # if '2-1' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '2-3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '2-5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '2-7' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '2-9' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        #
        # elif '6-1' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '6-3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '6-5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '6-7' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '6-9' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        #

        # if 'above3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'above4' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'above5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'ahead3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'ahead4' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'ahead5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        #
        # elif 'aheadabove3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'aheadabove4' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'aheadabove5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'behind3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'behind4' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'behind5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        #
        # elif 'below3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'below4' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'below5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        #
        # elif 'left3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'left4' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'left5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        #
        # elif 'right3' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'right4' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif 'right5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)

        # if '2-5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '3-7' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '4-7' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '5-7' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
        # elif '6-5' in img_s:
        #     img_path = os.path.join(file_h, img)
        #     dst_dir = '/home/moktari/Moktari/CITeR_2022/M2FPA/m2fpa/'
        #     new_name = file + '_' + img_s + 'n' + '.jpg'
        #     dst_path = os.path.join(dst_dir, new_name)
        #     shutil.copy(img_path, dst_path)
