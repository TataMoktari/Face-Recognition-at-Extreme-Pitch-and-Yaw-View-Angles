import os
data_dir = "/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition_joint_Pitch_Yaw/Pitch_15/Pitch_pos_15/Gallery"
p = open(
    "/home/moktari/Moktari/2022/facenet-pytorch-master/Facenet_Finetune_Pose_aware_Attention/M2FPA/Rank_1_Recognition_joint_Pitch_Yaw/Pitch_15/Pitch_pos_15/Gallery_list.txt",
    'w')

for file1 in sorted(os.listdir(data_dir)):
    filepath = os.path.join(data_dir, file1)
    p.write(filepath + '\n')
p.close()
