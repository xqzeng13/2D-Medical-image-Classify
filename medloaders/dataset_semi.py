import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medical_loader_utils1 import  new_read_Exist_volumes
class Dataset_unlabeled(Dataset):
    def __init__(self,list):
        self.list = []
        self.list = list#new_read_Exist_volume
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):

        ####todo

        t1_path, seg_path= self.list[index]####加载第一组npy文件11849
        # print("unlabeled:",t1_path)
        # print(self.list[index])
        t1= np.load(t1_path)
        s=np.zeros(t1.shape)
        # if self.mode == 'train' and self.augmentation:
        #     print('augmentation reee')
        #     [augmented_t1], augmented_s = self.transform([t1], s)
        #
        #     return torch.FloatTensor(augmented_t1.copy()).unsqueeze(0),torch.FloatTensor(augmented_s.copy())
        # img=sitk.ReadImage(t1_path)
        # mask=sitk.ReadImage(seg_path)
        # t1 = sitk.GetArrayFromImage(img)
        # s = sitk.GetArrayFromImage(mask)

        return torch.FloatTensor(t1).unsqueeze(0),  torch.FloatTensor(s)