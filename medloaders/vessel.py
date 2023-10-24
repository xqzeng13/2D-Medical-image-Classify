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


# from test import Test_Datasets

class vessel(Dataset):
    """
    Code for reading the infant brain MRI dataset of ISEG 2017 challenge
    """

    def __init__(self, args, mode):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param fold_id: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(args.pathname)
        # self.training_path = args.trainpath
        # self.label_path = args.labelpath
        # self.testing_path = args.testpath
        # self.CLASSES = args.classes
        # self.full_vol_dim = args.full_vol_dim # slice, width, height
        # self.threshold = args.threshold
        # self.normalization = args.normalization
        # self.augmentation = args.augmentation
        # self.crop_size = crop_dim
        self.list = []
        # self.samples = samples
        # self.samples1 = samples1
        # self.full_volume = None
        # self.save_name = self.root +'/' +args.dataset_name+'/Traininglist-' + mode + '-samples-' + str(
        #     1000) + '.txt'
        ###数据扩充######
        # if self.augmentation:######default False
        #     self.transform = augment3D.RandomChoice(
        #         transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
        #                     augment3D.ElasticTransform()], p=0.5)
        # if load:
        #     ## load pre-generated data
        #     # self.list = utils.load_list(self.save_name)
        #     list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*.nii.gz')))
        #     labels = sorted(glob.glob(os.path.join(self.label_path, '*.nii.gz')))
        #     self.affine = img_loader.load_affine_matrix(list_IDsT1[0])
        #     return list_IDsT1, labels, self.affine
        #
        # else:
        #     subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
        #     self.sub_vol_path = self.root + '/' +args.dataset_name+ '/generated/' + mode + subvol + '/'
        #
        #     utils.make_dirs(self.sub_vol_path)
        #     list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*.nii.gz')))
        #     # print(list_IDsT1)
        #
        #     labels = sorted(glob.glob(os.path.join(self.label_path, '*.nii.gz')))
        #     self.affine = img_loader.load_affine_matrix(list_IDsT1[0])



        if self.mode =='train_exist':
            # self.list = read_Exist_volumes(mode=mode)#
            self.list = new_read_Exist_volumes(mode=mode,csvname=args.csvname,pathname=args.pathname)#new_read_Exist_volumes

        elif self.mode =='val_exist':
            # self.list=read_Exist_volumes(mode=mode)
            self.list = new_read_Exist_volumes(mode=mode,csvname=args.csvname,pathname=args.pathname)#new_read_Exist_volumes
        # utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        ####todo

        t1_path, seg_path= self.list[index]####加载第一组npy文件
        # print(self.list[index])
        t1, s = np.load(t1_path), np.load(seg_path)
        # if self.mode == 'train' and self.augmentation:
        #     print('augmentation reee')
        #     [augmented_t1], augmented_s = self.transform([t1], s)
        #
        #     return torch.FloatTensor(augmented_t1.copy()).unsqueeze(0),torch.FloatTensor(augmented_s.copy())

        return torch.FloatTensor(t1).unsqueeze(0),  torch.FloatTensor(s)

class Dataset(Dataset):
    def __init__(self,list):
        self.list = []
        self.list = list#new_read_Exist_volume
    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):

        ####todo

        t1_path, seg_path= self.list[index]####加载第一组npy文件11849
        # print(self.list[index])
        # print("labeled:",t1_path)

        t1 = np.load(t1_path)
        s= np.load(seg_path)
        # if self.mode == 'train' and self.augmentation:
        #     print('augmentation reee')
        #     [augmented_t1], augmented_s = self.transform([t1], s)
        #
        #     return torch.FloatTensor(augmented_t1.copy()).unsqueeze(0),torch.FloatTensor(augmented_s.copy())
        # img=sitk.ReadImage(t1_path)
        # mask=sitk.ReadImage(seg_path)
        # t1 = sitk.GetArrayFromImage(img)
        # s = sitk.GetArrayFromImage(mask)
        # print(self.list[index])
        return torch.FloatTensor(t1).unsqueeze(0),  torch.FloatTensor(s)
