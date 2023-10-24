from torch.utils.data import DataLoader

from .COVIDxdataset import COVIDxDataset
from .Covid_Segmentation_dataset import COVID_Seg_Dataset
from .brats2018 import MICCAIBraTS2018
from .brats2019 import MICCAIBraTS2019
from .brats2020 import MICCAIBraTS2020
from .covid_ct_dataset import CovidCTDataset
from .iseg2017 import MRIDatasetISEG2017
from .iseg2019 import MRIDatasetISEG2019
from .ixi_t1_t2 import IXIMRIdataset
from .miccai_2019_pathology import MICCAI2019_gleason_pathology
# from .mrbrains2018 import MRIDatasetMRBRAINS2018
from.vessel import vessel,Dataset
from .dataset_semi import Dataset_unlabeled
# from.new_vessel import new_vessel
import os
import pandas as pd
import random


def newgenerate_datasets(args):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 64,
              'pin_memory':False,
              # 'multiprocessing_context':'spawn'
              }
    train_csv=r'/data4/zengxq/datas109_45/109/64x64x32/trian.csv'
    val_csv=r'/data4/zengxq/datas109_45/109/64x64x32/val.csv'

    # aug_train_path=r'/data4/zengxq/datas109_45/45/Patches/64/'
    patch_label=r'/data4/zengxq/datas109_45/109/64x64x32/train_val_64/'
    patch_val=r'/data4/zengxq/datas109_45/109/64x64x32/train_val_64/'
    train_list = read_train_val(train_csv, patch_label)
    val_list = read_train_val(val_csv, patch_val)

    print("train patch sum is ",len(train_list),'\nval patch sum is' ,len(val_list))
    train_loader = Dataset(train_list)
    val_loader = Dataset(val_list)

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator
def newgenerate_datasets_pseudo(args):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 64,
              'pin_memory':False,
              # 'multiprocessing_context':'spawn'
              }

    train_csv=r'/data4/zengxq/SSL/train_labeled.csv'
    train_pseudo_csv=r'/data4/zengxq/SSL/train_unlabeled.csv'
    val_csv=r'/data4/zengxq/SSL/val.csv'
    # aug_train_path=r'/data4/zengxq/datas109_45/45/Patches/64/'
    patch_label=r'/data4/zengxq/SSL/patch_labeled/'
    patch_unlabeled=r'/data4/zengxq/SSL/patch_unlabeled/'
    patch_val=r'/data4/zengxq/SSL/patch_val/'

    train_label_list = read_train_val(train_csv, patch_label)
    train_unlabel_list=read_train_val_pseudo(train_pseudo_csv,patch_unlabeled,patch_unlabeled)
    val_list = read_train_val(val_csv, patch_val)


    train_list=train_label_list+train_unlabel_list
    # train_list=train_label_list


    print("train patch sum is ",len(train_list),'\nval patch sum is' ,len(val_list))
    train_loader = Dataset(train_list)
    val_loader = Dataset(val_list)

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator
def newgenerate_datasets_classify(args):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 0,
              'pin_memory':False,
              # 'multiprocessing_context':'spawn'
              }
    # samples_train = args.samples_train
    # samples_val = args.samples_val
    # split_percent = args.split
    # # total_data=len(os.listdir(args.trainpath))
    # #
    # split_idx = 1
    # train_csv = read_Exist_volumes(aug_train_path)
    # train_val_csv=r'/data4/zengxq/datas109_45/109/64train_val.csv'
    train_csv=r'/data4/zengxq/XF_PET/train.csv'

    val_csv=r'/data4/zengxq/XF_PET/val.csv'

    # aug_train_path=r'/data4/zengxq/datas109_45/45/Patches/64/'
    patch_label=r'/data4/zengxq/XF_PET/patch/'
    # patch_unlabel_data=r'/data4/zengxq/SSL/patch_unlabeled/'
    # patch_unlabel_pseudo=r'/data4/zengxq/SSL/patch_unlabeled_pred/'


    # patch_val=r'/data4/zengxq/SSL/patch_val/'
    # split=0.8
    # train_list, val_list = split_train_val(train_val_csv, aug_train_path, split)
    # T=100000
    # V=15000
    # train_label_list = read_train_val(train_csv, patch_label)
    # train_unlabel_list = read_train_val_pseudo(train_pseudo_csv, patch_unlabel_data,patch_unlabel_pseudo)
    train_list=read_train_val(train_csv, patch_label)
    val_list = read_train_val(val_csv, patch_label)

    print("train patch sum is ",len(train_list),'\nval patch sum is' ,len(val_list))
    train_loader = Dataset(train_list)
    val_loader = Dataset(val_list)

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator
def newgenerate_datasets_semi(args):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 64,
              'pin_memory':False,
              # 'multiprocessing_context':'spawn'
              }
    #idx_file
    train_labelcsv=r'/data4/zengxq/SSL/patch_labeled.csv'
    train_unlabelcsv=r'/data4/zengxq/SSL/patch_labeled.csv'
    val_csv=r'/data4/zengxq/SSL/patch_val.csv'
    #file_path
    patch_label=r'/data4/zengxq/SSL/patch_labeled/'
    patch_unlabel=r'/data4/zengxq/SSL/patch_labeled/'#unlabe
    patch_val=r'/data4/zengxq/SSL/patch_val/'

    train_label_list = read_train_val(train_labelcsv, patch_label)
    train_unlabel_list=read_train_val(train_unlabelcsv, patch_unlabel)
    val_list = read_train_val(val_csv, patch_val)

    print("train labeled patch sum is ",len(train_label_list),'\ntrain unlabel patch sum is' ,len(train_unlabel_list),'\nval patch sum is' ,len(val_list))
    train_labelloader = Dataset(train_label_list)
    train_unlabelloader = Dataset_unlabeled(train_unlabel_list)
    val_loader = Dataset(val_list)

    training_label_generator = DataLoader(train_labelloader, **params)
    training_unlabel_generator = DataLoader(train_unlabelloader, **params)

    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_label_generator,training_unlabel_generator, val_generator
# def read_Exist_volumes(aug_path):
#     filenamelist = []
#     classificationlist = []
#     train_csv = os.path.join(aug_path + '/patch.csv')
#     datanameList = sorted(glob(os.path.join(aug_path, '*.npy')))
#     for i in tqdm(range(len(datanameList))):
#         name = datanameList[i].split('aug_train/')[-1].replace('.npy', '')
#         label = name.split('_T_')[1][0:1]
#         if label == 'i':
#             label = label.replace('i', '4')
#         else:
#             label = label
#         filenamelist.append(name)
#         classificationlist.append(label)
#         output_excel = {'id': [], 'label': []}
#         output_excel['id'] = filenamelist
#         output_excel['label'] = classificationlist
#         output = pd.DataFrame(output_excel)
#         output.to_csv(train_csv, index=False)
#     return train_csv
def read_train_val(csv_path, aug_path):
    patch_list = []
    val_list = []
    all_list = []
    csvname = csv_path
    pathname = aug_path
    # x=x
    input_df = pd.read_csv(csvname)
    print("\n sum of  patches is :", input_df.shape[0])
    for i in range(input_df.shape[0]):
        # dataname = pathname + input_df.iloc[i].at['filename']
        labelname = pathname + input_df.iloc[i].at['filename']
        # labelname=input_df.iloc[i].at['number']
        dataname = labelname.replace('seg.npy','0.npy')
        # labelname = input_df.iloc[i].at['filename']
        # dataname =  str(input_df.iloc[i].at['filename']).split('seg')[0] + '0.npy'
        tuples = (dataname, labelname)
        patch_list.append(tuples)
        print('\r[ %d / %d]' % (i, input_df.shape[0]), end='')
    #
    # random.shuffle(all_list)
    # train_list = all_list[:int(input_df.shape[0] * split_idx)]
    # val_list = all_list[int(input_df.shape[0] * split_idx):]
    print("\r data_split  is finish!")

    return patch_list
def read_train_val_pseudo(csv_path, data_path,pseudo_path):
    patch_list = []
    val_list = []
    all_list = []
    csvname = csv_path
    # data = data_path
    # pseudo=pseudo_path
    # x=x
    input_df = pd.read_csv(csvname)
    print("\n sum of  pseudo patches is :", input_df.shape[0])
    for i in range(input_df.shape[0]):
        dataname = pseudo_path + input_df.iloc[i].at['filename']
        labelname =dataname.replace('0.npy','gt.npy')
        # labelname = input_df.iloc[i].at['filename']
        # dataname =  str(input_df.iloc[i].at['filename']).split('seg')[0] + '0.npy'
        tuples = (dataname, labelname)
        patch_list.append(tuples)
        print('\r[ %d / %d]' % (i, input_df.shape[0]), end='')
    #
    # random.shuffle(all_list)
    # train_list = all_list[:int(input_df.shape[0] * split_idx)]
    # val_list = all_list[int(input_df.shape[0] * split_idx):]
    print("\r data_split  is finish!")

    return patch_list
def generate_datasets(args, path='.././datasets'):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    samples_train = args.samples_train
    samples_val = args.samples_val
    split_percent = args.split

    if args.dataset_name == "vessel":
        total_data = 10
        split_idx = int(split_percent * total_data)
        train_loader = MRIDatasetISEG2017(args, 'train', dataset_path=path, crop_dim=args.dim,
                                          split_id=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetISEG2017(args, 'val', dataset_path=path, crop_dim=args.dim, split_id=split_idx,
                                        samples=samples_val, load=args.loadData)

    if args.dataset_name == "iseg2017":
        total_data = 10
        split_idx = int(split_percent * total_data)
        train_loader = MRIDatasetISEG2017(args, 'train', dataset_path=path, crop_dim=args.dim,
                                          split_id=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetISEG2017(args, 'val', dataset_path=path, crop_dim=args.dim, split_id=split_idx,
                                        samples=samples_val, load=args.loadData)

    elif args.dataset_name == "iseg2019":
        total_data = 10
        split_idx = int(split_percent * total_data)
        train_loader = MRIDatasetISEG2019(args, 'train', dataset_path=path, crop_dim=args.dim,
                                          split_id=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetISEG2019(args, 'val', dataset_path=path, crop_dim=args.dim, split_id=split_idx,
                                        samples=samples_val, load=args.loadData)
    elif args.dataset_name == "mrbrains4":
        train_loader = MRIDatasetMRBRAINS2018(args, 'train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              split_id=0, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetMRBRAINS2018(args, 'val', dataset_path=path, classes=args.classes, dim=args.dim,
                                            split_id=0,
                                            samples=samples_val, load=args.loadData)
    elif args.dataset_name == "mrbrains9":
        train_loader = MRIDatasetMRBRAINS2018(args, 'train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              split_id=0, samples=samples_train, load=args.loadData)

        val_loader = MRIDatasetMRBRAINS2018(args, 'val', dataset_path=path, classes=args.classes,
                                            dim=args.dim,
                                            split_id=0,
                                            samples=samples_val, load=args.loadData)
    elif args.dataset_name == "miccai2019":
        total_data = 244
        split_idx = int(split_percent * total_data) - 1

        val_loader = MICCAI2019_gleason_pathology(args, 'val', dataset_path=path, split_idx=split_idx,
                                                  crop_dim=args.dim,
                                                  classes=args.classes, samples=samples_val,
                                                  save=True)

        print('Generating train set...')
        train_loader = MICCAI2019_gleason_pathology(args, 'train', dataset_path=path, split_idx=split_idx,
                                                    crop_dim=args.dim,
                                                    classes=args.classes, samples=samples_train,
                                                    save=True)

    elif args.dataset_name == "ixi":
        loader = IXIMRIdataset(args, dataset_path=path, voxels_space=args.dim, modalities=args.inModalities, save=True)
        generator = DataLoader(loader, **params)
        return generator, loader.affine

    elif args.dataset_name == "brats2018":
        total_data = 244
        split_idx = int(split_percent * total_data)
        train_loader = MICCAIBraTS2018(args, 'train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MICCAIBraTS2018(args, 'val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val, load=args.loadData)

    elif args.dataset_name == "brats2019":
        split = (0.8, 0.2)
        total_data = 335
        split_idx = int(split[0] * total_data)
        train_loader = MICCAIBraTS2019(args, 'train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MICCAIBraTS2019(args, 'val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val, load=args.loadData)

    elif args.dataset_name == "brats2020":
        split = (0.8, 0.2)
        total_data = 335
        split_idx = int(split[0] * total_data)
        train_loader = MICCAIBraTS2020(args, 'train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train, load=args.loadData)

        val_loader = MICCAIBraTS2020(args, 'val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val, load=args.loadData)
    elif args.dataset_name == 'COVID_CT':
        train_loader = CovidCTDataset('train', root_dir='.././datasets/covid_ct_dataset/',
                                      txt_COVID='.././datasets/covid_ct_dataset/trainCT_COVID.txt',
                                      txt_NonCOVID='.././datasets/covid_ct_dataset/trainCT_NonCOVID.txt')
        val_loader = CovidCTDataset('val', root_dir='.././datasets/covid_ct_dataset',
                                    txt_COVID='.././datasets/covid_ct_dataset/valCT_COVID.txt',
                                    txt_NonCOVID='.././datasets/covid_ct_dataset/valCT_NonCOVID.txt')
    elif args.dataset_name == 'COVIDx':
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=path,
                                     dim=(224, 224))
        val_loader = COVIDxDataset(mode='val', n_classes=args.classes, dataset_path=path,
                                   dim=(224, 224))

    elif args.dataset_name == 'covid_seg':
        train_loader = COVID_Seg_Dataset(mode='train', dataset_path=path, crop_dim=args.dim,
                                         fold=0, samples=samples_train)

        val_loader = COVID_Seg_Dataset(mode='val', dataset_path=path, crop_dim=args.dim,
                                       fold=0, samples=samples_val)
    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator, val_loader.full_volume, val_loader.affine


def select_full_volume_for_infer(args, path='.././datasets'):
    params = {'batch_size': args.batchSz,
              'shuffle': True,
              'num_workers': 2}
    samples_train = args.samples_train
    samples_val = args.samples_val
    split_percent = args.split

    if args.dataset_name == "iseg2017":
        total_data = 10
        split_idx = int(split_percent * total_data)
        loader = MRIDatasetISEG2017('viz', dataset_path=path, crop_dim=args.dim,
                                    split_id=split_idx, samples=samples_train)


    elif args.dataset_name == "iseg2019":
        total_data = 10
        split_idx = int(split_percent * total_data)
        train_loader = MRIDatasetISEG2019('train', dataset_path=path, crop_dim=args.dim,
                                          split_id=split_idx, samples=samples_train)

        val_loader = MRIDatasetISEG2019('val', dataset_path=path, crop_dim=args.dim, split_id=split_idx,
                                        samples=samples_val)
    elif args.dataset_name == "mrbrains4":
        train_loader = MRIDatasetMRBRAINS2018('train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              split_id=0, samples=samples_train)

        val_loader = MRIDatasetMRBRAINS2018('val', dataset_path=path, classes=args.classes, dim=args.dim,
                                            split_id=0,
                                            samples=samples_val)
    elif args.dataset_name == "mrbrains9":
        train_loader = MRIDatasetMRBRAINS2018('train', dataset_path=path, classes=args.classes, dim=args.dim,
                                              split_id=0, samples=samples_train)

        val_loader = MRIDatasetMRBRAINS2018('val', dataset_path=path, classes=args.classes,
                                            dim=args.dim,
                                            split_id=0,
                                            samples=samples_val)
    elif args.dataset_name == "miccai2019":
        total_data = 244
        split_idx = int(split_percent * total_data) - 1

        val_loader = MICCAI2019_gleason_pathology('val', dataset_path=path, split_idx=split_idx, crop_dim=args.dim,
                                                  classes=args.classes, samples=samples_val,
                                                  save=True)

        print('Generating train set...')
        train_loader = MICCAI2019_gleason_pathology('train', dataset_path=path, split_idx=split_idx, crop_dim=args.dim,
                                                    classes=args.classes, samples=samples_train,
                                                    save=True)

    elif args.dataset_name == "ixi":
        loader = IXIMRIdataset(dataset_path=path, voxels_space=args.dim, modalities=args.inModalities, save=True)
        generator = DataLoader(loader, **params)
        return generator, loader.affine

    elif args.dataset_name == "brats2018":
        total_data = 244
        split_idx = int(split_percent * total_data)
        train_loader = MICCAIBraTS2018('train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train)

        val_loader = MICCAIBraTS2018('val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val)

    elif args.dataset_name == "brats2019":
        split = (0.8, 0.2)
        total_data = 335
        split_idx = int(split[0] * total_data)
        train_loader = MICCAIBraTS2018('train', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                       split_idx=split_idx, samples=samples_train)

        val_loader = MICCAIBraTS2018('val', dataset_path=path, classes=args.classes, crop_dim=args.dim,
                                     split_idx=split_idx,
                                     samples=samples_val)
    elif args.dataset_name == 'COVID_CT':
        train_loader = CovidCTDataset('train', root_dir='.././datasets/covid_ct_dataset/',
                                      txt_COVID='.././datasets/covid_ct_dataset/trainCT_COVID.txt',
                                      txt_NonCOVID='.././datasets/covid_ct_dataset/trainCT_NonCOVID.txt')
        val_loader = CovidCTDataset('val', root_dir='.././datasets/covid_ct_dataset',
                                    txt_COVID='.././datasets/covid_ct_dataset/valCT_COVID.txt',
                                    txt_NonCOVID='.././datasets/covid_ct_dataset/valCT_NonCOVID.txt')
    elif args.dataset_name == 'COVIDx':
        train_loader = COVIDxDataset(mode='train', n_classes=args.classes, dataset_path=path,
                                     dim=(224, 224))
        val_loader = COVIDxDataset(mode='val', n_classes=args.classes, dataset_path=path,
                                   dim=(224, 224))

    elif args.dataset_name == 'covid_seg':
        train_loader = COVID_Seg_Dataset(mode='train', dataset_path=path, crop_dim=args.dim,
                                         fold=0, samples=samples_train)

        val_loader = COVID_Seg_Dataset(mode='val', dataset_path=path, crop_dim=args.dim,
                                       fold=0, samples=samples_val)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return loader.full_volume, loader.affine
