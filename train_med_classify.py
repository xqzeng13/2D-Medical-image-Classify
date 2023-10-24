# Python libraries
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
import torch
import torch.nn as nn
# Lib files
import lib.medloaders as medical_loaders
from models import create_model
import lib.train as train
import lib.utils as utils
from lib.losses3D import DiceLoss
from lib.losses3D import BCEDiceLoss
import torch.nn as nn
from torch.nn import DataParallel
import torch.multiprocessing as mp
# torch.multiprocessing.set_sharing_strategy('file_system')#file_system /file_descriptor
# from lib.loss_function.losses import binary_focal_loss
# from lib.loss_function.losses import Focal_Loss
from lib.loss_function.focal_loss import FocalLoss
from lib.loss_function.focal_loss import CELoss
from lib.loss_function.focal_loss import BCEloss

# from lib.losses3D import FocalLoss

from  lib.losses3D.loss_function import Binary_Loss
from lib.losses3D import weight_cross_entropy
from lib.losses3D import focal_loss as focal_loss
# import multiprocessing as mp
# mp.set_start_method('spawn', force=True)
# mp.set_sharing_strategy('file_system')
# ctx = mp.get_context()
# ctx.shared_memory_timeout = 300  # 增加超时时间为 300 秒

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"#####"0"表示用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# gpus = [3,4,5,6]
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
seed = 1777777
# 指定使用的GPU设备


def main():
    args = get_arguments()
    ##TODO 是否继续已有模型基础上训练：True为是
    model_save="False"#False/True
    model_save_path=r"/home/lisic/zengxq/finnal_vessel_project/saved_models/"
    # a = torch.cuda.is_available()###判断是否使用GPU
    # print(a)

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    # training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
    #                                                                                            path='.././datasets')
    training_generator, val_generator = medical_loaders.newgenerate_datasets_classify(args,)
    # training_generator, val_generator = None,None

    model, optimizer = create_model(args)
    if  model_save=="True":
        ckpt = torch.load(
            r'/home/lisic/zengxq/finnal_vessel_project/saved_models/dwi_newUNET3D_checkpoints/UNET3D_02_04___06_21_vessel_/UNET3D_02_04___06_21_vessel__BEST.pth')  # model_path.path
        model.load_state_dict(ckpt)
    else :
        model=model
    #多GPU训练
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.cuda.set_device(device_ids[0])
    # model = nn.DataParallel(model, device_ids=device_ids)
    #     # print(torch.cuda.device_count())
    # model.to(device_ids[0])
    criterion = BCEDiceLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.to(device)
    # model = DataParallel(model)###！！！！！！！！！！！！！！！！！！0313
    # 查看device的属性：GPU个数，用于检查是否gpu设置是否正确

    #     model = nn.DataParallel(model)
    # 模型导入GPU

    # if args.cuda:
    #     model = model.cuda()
    #     print("Model transferred in GPU.....")
    ###################
    # 将模型移动到GPU上

    # 打印模型的参数和缓冲区所在的设备
    # for name, param in model.named_parameters():
    #     print(name, param.device)
    # # 将模型封装在数据并行容器中
    # if torch.cuda.is_available():
    #     # torch.cuda.set_device(device_ids)
    #     model = nn.DataParallel(model, device_ids=device_ids)
######################
    trainformat="classifyk3"
    trainer = train.Trainer_classify(args, model, device,trainformat,criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, )
    print("START TRAINING...")
    trainer.training()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchSz', type=int, default=100)#64*64*64 5gpus 34000Mb
    parser.add_argument('--full_vol_dim',default=(448,448,128))
    parser.add_argument('--dataset_name', type=str, default="vessel")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64,32))
    parser.add_argument('--nEpochs', type=int, default=500)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--samples_train', type=int, default=2)
    parser.add_argument('--samples_val', type=int, default=1)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--csvname',  default=r'/data4/zengxq/lung/64x64x32/')
    parser.add_argument('--pathname',  default=r'/data4/zengxq/lung/64x64x32/train_val_ord/')
    # parser.add_argument('--testpath', default=r'/data4/zengxq/GoldNormaldatas20/')
    parser.add_argument('--threshold', default=0.004, type=float)
    parser.add_argument('--terminal_show_freq', default=100)
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--split', default=0.8, type=float, help='Select percentage of training data(default: 0.8)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate (default: 1e-3)')##
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--loadData', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='ResNetclassify',
                        choices=(
                        'UNETR','VoxResnet','UARAI_classify','ResNetclassify','VNET',"VNET_new",'UARAI', 'UNET3D', 'UNet3D_attention', 'UNet3D_Rattention', 'UNet3D_Rattention_edgedetect', 'RESNETMED3D'

                        , 'UNET3D', 'UNet3D_attention', 'UNet3D_attention_addjuchi', 'UNet3D_attention_inception'
                        , 'UNet3D_Rattention', 'UNet3D_Rattention_juchi', 'UNet3D_Rattention_new',
                        'UNet3D_Rattention_inception'))
    parser.add_argument('--target', type=str, default='ResNetclassify18',
                        choices=('vessel',"airway"))

    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--log_dir', type=str,
                        default='../runs/')

    args = parser.parse_args()

    args.save = '../saved_models/' +args.target+ args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    return args


if __name__ == '__main__':

    main()
