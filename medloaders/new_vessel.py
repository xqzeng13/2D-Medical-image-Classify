import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medical_loader_utils1 import get_viz_set,  create_order_train_direct_volumes, \
    create_order_val_direct_volumes, create_order_test_direct_volumes
# from lib.medical_loader_utils1 import
########生成pathsh的函数

class new_vessel(Dataset):
    """
    Code for reading the infant brain MRI dataset of ISEG 2017 challenge
    """

    def __init__(self, args, mode, dataset_path='./datasets', crop_dim=(16, 16, 16), split_id=1, samples=1000,
                 load=False):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        :param crop_dim: subvolume tuple
        :param fold_id: 1 to 10 values
        :param samples: number of sub-volumes that you want to create
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.training_path = args.trainpath
        self.label_path = args.labelpath
        self.testing_path = args.testpath
        self.CLASSES = args.classes
        self.full_vol_dim = args.full_vol_dim  # slice, width, height
        self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.crop_size = crop_dim
        self.list = []
        self.samples = samples
        # self.test_dsc =args.test_dsc
        # self.samples1 = samples1
        self.full_volume = None
        self.save_name = self.root + '/' + args.dataset_name + '/Traininglist-' + mode + '-samples-' + str(
            samples) + '.txt'
        ###数据扩充######

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2])
        self.sub_vol_path = self.root + '/' + args.dataset_name + '/generated/' + mode + subvol + '/'
        self.part_vol_path = self.root + '/' + args.dataset_name + '/generated1%/' + mode + subvol + '/'
        utils.make_dirs(self.sub_vol_path)

        utils.make_dirs(self.part_vol_path)
        # list_IDsT1 = sorted(glob.glob(os.path.join(self.training_path, '*.nii.gz')))
        # print(list_IDsT1)
        traindatapath = r'D:\Work\Datasets\Data_augmentation\new_datasets\train\data\\'
        valdatapath=r'D:\Work\Datasets\Data_augmentation\val\data\\'
        testdatapath=r'D:\Work\Datasets\GoldNormaldatas20\test2\data\\'
        #
        # labels = sorted(glob.glob(os.path.join(self.label_path, '*.nii.gz')))
        # self.affine = img_loader.load_affine_matrix(list_IDsT1[0])
        trainlabelpath = r'D:\Work\Datasets\Data_augmentation\new_datasets\train\label\\'
        vallabelpath=r'D:\Work\Datasets\Data_augmentation\val\label\\'
        testlabelpath=r'D:\Work\Datasets\GoldNormaldatas20\test2\label\\'

        if self.mode == 'train':

            # list_IDsT1 = list_IDsT1[:split_id]
            #
            # labels = labels[:split_id]
            ###随机取patch#####
            # self.list = create_train_sub_volumes(trainpath, labelpath, dataset_name=args.dataset_name,
            #                                      mode=mode, samples=samples, full_vol_dim=self.full_vol_dim,
            #                                      crop_size=self.crop_size,
            #                                      sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
            #                                      normalization=args.normalization)

            ##顺序取patch#####
            # self.list_two = create_order_train_sub_volumes(list_IDsT1, labels, dataset_name=args.dataset_name,
            #                                      mode=mode, samples=samples, full_vol_dim=self.full_vol_dim,
            #                                      crop_size=self.crop_size,
            #                                      sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
            #                                      normalization=args.normalization)

            # ###讲随机取得和顺序取得patch存入list
            # self.list=self.list_two[0]
            # self.listpart=self.list_two[1]
            # print(self.list)
            # print(self.listpart)

            # self.list = random_list.append(order_list)
            self.list = create_order_train_direct_volumes(traindatapath, trainlabelpath, dataset_name=args.dataset_name,
                                                          mode=mode, samples=samples, full_vol_dim=self.full_vol_dim,
                                                          crop_size=self.crop_size,
                                                          sub_vol_path=self.sub_vol_path,
                                                          part_vol_path=self.part_vol_path,
                                                          th_percent=self.threshold,
                                                          normalization=args.normalization)

        # elif self.mode == 'train_exist':
        #     self.list = read_Exist_volumes(model=mode)

        # elif self.mode=='test_exist':
        #     self.list=T

        elif self.mode == 'val':
            utils.make_dirs(self.sub_vol_path)

            self.list = create_order_val_direct_volumes(valdatapath, vallabelpath, dataset_name=args.dataset_name,
                                               mode=mode, samples=samples, full_vol_dim=self.full_vol_dim,
                                               crop_size=self.crop_size,
                                               sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                                part_vol_path=self.part_vol_path,
                                               normalization=args.normalization)

            self.full_volume = get_viz_set(valdatapath, vallabelpath, dataset_name=args.dataset_name)
        elif self.mode == 'test':
            utils.make_dirs(self.sub_vol_path)

            self.list = create_order_test_direct_volumes(testdatapath, testlabelpath, dataset_name=args.dataset_name,
                                                        mode=mode, samples=samples, full_vol_dim=self.full_vol_dim,
                                                        crop_size=self.crop_size,
                                                        sub_vol_path=self.sub_vol_path, th_percent=self.threshold,
                                                        part_vol_path=self.part_vol_path,
                                                        normalization=args.normalization,test_dsc=self.test_dsc)

            self.full_volume = get_viz_set(valdatapath, vallabelpath, dataset_name=args.dataset_name)

        utils.save_list(self.save_name, self.list)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        t1_path, seg_path = self.list[index]  ####加载第一组npy文件
        t1, s = np.load(t1_path), np.load(seg_path)
        if self.mode == 'train' and self.augmentation:
            print('augmentation reee')
            [augmented_t1], augmented_s = self.transform([t1], s)

            return torch.FloatTensor(augmented_t1.copy()).unsqueeze(0), torch.FloatTensor(augmented_s.copy())

        return torch.FloatTensor(t1).unsqueeze(0), torch.FloatTensor(s)
