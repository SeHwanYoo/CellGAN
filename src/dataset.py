import os
import random
from skimage import img_as_float32
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

categories = {
    'Artefacts':    [1, 0, 0, 0],
    'Low_Contrast': [0, 1, 0, 0],
    'Low_Texture':  [0, 0, 1, 0],
    'Out_of_Focus': [0, 0, 0, 1],
}


# One-hot class labels
# categories = {
#     'NILM':   [1, 0, 0, 0, 0],
#     'ASC_US': [0, 1, 0, 0, 0],
#     'LSIL':   [0, 0, 1, 0, 0],
#     'ASC_H':  [0, 0, 0, 1, 0],
#     'HSIL':   [0, 0, 0, 0, 1],
# }

# categories = {
#     '0':  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     '1':  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     '2':  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     '3':  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     '4':  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     '5':  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     '6':  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#     '7':  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#     '8':  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#     '9':  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#     '10': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#     '11': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#     '12': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#     '13': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#     '14': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
# }

# categories = {
#     'AGC':    [1, 0, 0, 0, 0, 0, 0],
#     'ASC-H':  [0, 1, 0, 0, 0, 0, 0],
#     'ASC-US': [0, 0, 1, 0, 0, 0, 0],
#     'GEC':    [0, 0, 0, 1, 0, 0, 0],
#     'HSIL':   [0, 0, 0, 0, 1, 0, 0],
#     'LSIL':   [0, 0, 0, 0, 0, 1, 0],
#     'NILM':   [0, 0, 0, 0, 0, 0, 1],
# }

# categories = {
#     'Normal':    [1, 0, 0, 0, 0],
#     'G3':        [0, 1, 0, 0, 0],
#     'G4':        [0, 0, 1, 0, 0],
#     'G5':        [0, 0, 0, 1, 0],
#     'Stroma':    [0, 0, 0, 0, 1],
# }

# categories = {
#     'AGC':    [0.5, 0.5, 0, 0, 0, 0, 0],
#     'ASC-H':  [0, 0.5, 0.5, 0, 0, 0, 0],
#     'ASC-US': [0, 0, 0.5, 0.5, 0, 0, 0],
#     'GEC':    [0, 0, 0, 0.5, 0.5, 0, 0],
#     'HSIL':   [0, 0, 0, 0, 0.5, 0.5, 0],
#     'LSIL':   [0, 0, 0, 0, 0, 0.5, 0.5],
#     'NILM':   [0.5, 0, 0, 0, 0, 0, 0.5],
# }



class CellDataset(Dataset):
    def __init__(self, dataroot, normalize=False, augment=False, cls=None):
        super().__init__()

        # Assign available categories
        if cls is not None:          # EXAMPLE: cls="LSIL" | cls = "NILM,ASC_US,HSIL"
            for sub_dir in cls.split(','):
                assert sub_dir in list(categories.keys()), "Unsupported category: {:s}".format(sub_dir)
            self.available_categories = cls.split(',')
        else:
            self.available_categories = list(categories.keys())

        # Collect data
        self.data = []
        with open(os.path.join(dataroot, "Training_Patches.txt")) as file:
            for line in file:
                img_path = os.path.join(dataroot, line.strip()).replace('\\', '/')
                # category = line.split('/')[0]
                category = line.split('\\')[0]
                
                if category in self.available_categories:
                    class_index = categories[category]
                    self.data.append([img_path, class_index])
        random.shuffle(self.data)
        
        # print('---------------------->', len(self.data))

        # Transformation
        transform_list = []
        transform_list.append(transforms.Resize((256, 256)))
        if augment:
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())
        if normalize:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # range [-1, 1]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def _load_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [B, G, R] --> [R, G, B]
        img = img_as_float32(img)                   # range [0, 1]

        return img

    def __getitem__(self, index):
        # Load image and class label
        img_path, raw_label = self.data[index]
        img = self._load_img(img_path)

        # ndarray --> tensor
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        img = self.transform(img)
        label = torch.tensor(raw_label).contiguous().float()

        return img, label


def InfiniteSampler(n):
    """Data sampler"""
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSamplerWrapper(torch.utils.data.Sampler):
    """Data sampler wrapper"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
