import numpy as np
# from PIL.Image import Image

import PIL.Image as Image
from torch.utils.data import Dataset


class CMUPIE_SupConMemory(Dataset):
    def __init__(self, profile_df, frontal_df, transform=None, nce_k=200, num_positive=1, num_classes=7883):
        super(CMUPIE_SupConMemory, self).__init__()

        # self.mode = mode
        self.transform = transform
        self.k = nce_k
        self.num_positive = num_positive
        self.frontal_df = frontal_df
        self.profile_df = profile_df

        # profile ==============================================================================
        self.cls_positive_profile = [[] for i in range(num_classes)]

        num_samples = len(self.profile_df)
        for i in range(num_samples):
            self.cls_positive_profile[self.profile_df['id'][i]].append(i)

        self.cls_negative_profile = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative_profile[i].extend(self.cls_positive_profile[j])

        self.cls_positive_profile = [np.asarray(self.cls_positive_profile[i]) for i in range(num_classes)]
        self.cls_negative_profile = [np.asarray(self.cls_negative_profile[i]) for i in range(num_classes)]

        self.cls_positive_profile = np.asarray(self.cls_positive_profile)
        self.cls_negative_profile = np.asarray(self.cls_negative_profile)

        # profile ==============================================================================
        num_classes = max(self.frontal_df['id']) + 1
        self.cls_positive_frontal = [[] for i in range(num_classes)]
        num_samples = len(self.frontal_df)
        for i in range(num_samples):
            self.cls_positive_frontal[self.frontal_df['id'][i]].append(i)

        self.cls_negative_frontal = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative_frontal[i].extend(self.cls_positive_frontal[j])
        self.cls_positive_frontal = [np.asarray(self.cls_positive_frontal[i]) for i in range(num_classes)]
        self.cls_negative_frontal = [np.asarray(self.cls_negative_frontal[i]) for i in range(num_classes)]

        self.cls_positive_frontal = np.asarray(self.cls_positive_frontal)
        self.cls_negative_frontal = np.asarray(self.cls_negative_frontal)

    def __getitem__(self, item):

        profile, profile_lbl = Image.open(self.profile_df['path'][item]), self.profile_df['id'][item]

        pos_idx_frontal = np.random.choice(self.cls_positive_frontal[profile_lbl], 1)
        pos_idx_frontal = pos_idx_frontal[0]
        frontal = Image.open(self.frontal_df['path'][pos_idx_frontal])

        positive_frontal_idx = np.random.choice(self.cls_positive_frontal[profile_lbl], self.num_positive)
        # positive_frontal_idx = positive_frontal_idx
        positive_profile_idx = np.random.choice(self.cls_positive_profile[profile_lbl], self.num_positive)
        # positive_profile_idx = positive_profile_idx

        replace_frontal = True if self.k > len(self.cls_negative_frontal[profile_lbl]) else False
        neg_idx_frontal = np.random.choice(self.cls_negative_frontal[profile_lbl], self.k, replace=replace_frontal)
        if self.num_positive == 1:
            pos_neg_frontal = np.hstack((np.asarray([positive_frontal_idx[0]]), neg_idx_frontal))
        else:
            pos_neg_frontal = np.hstack((positive_frontal_idx, neg_idx_frontal))

        # sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        replace_profile = True if self.k > len(self.cls_negative_profile[profile_lbl]) else False
        neg_idx_profile = np.random.choice(self.cls_negative_profile[profile_lbl], self.k, replace=replace_profile)

        if self.num_positive == 1:
            pos_neg_profile = np.hstack((np.asarray([positive_profile_idx[0]]), neg_idx_profile))
        else:
            pos_neg_profile = np.hstack((positive_profile_idx, neg_idx_profile))

        label = 1
        if self.transform:
            profile = self.transform(profile)
            frontal = self.transform(frontal)

        return {'profile': profile,
                'frontal': frontal,
                'lbl': label,
                'id': profile_lbl,
                'neg_idx_frontal': pos_neg_frontal,
                'neg_idx_profile': pos_neg_profile,
                'y1': pos_idx_frontal,
                'y2': item}

    def __len__(self):
        return len(self.profile_df)

