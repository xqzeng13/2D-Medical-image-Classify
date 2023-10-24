import torch.nn as nn
import torch
from lib.losses3D.dice import DiceLoss
from lib.loss_function.losses import Focal_Loss
from lib.losses3D.basic import expand_as_one_hot
from sklearn.metrics import accuracy_score,auc,roc_curve
# Code was adapted and mofified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses3D"""

    def __init__(self, alpha=1, beta=1, classes=4):
        super(BCEDiceLoss, self).__init__()
        # self.alpha = alpha#####平衡因子，用于平衡正负比例
        # self.bce = nn.BCEWithLogitsLoss()
        # self.beta = beta
        # self.dice = DiceLoss(classes=classes)
        # self.classes=classes
        self.alpha = 1  #####平衡因子，用于平衡正负比例
        self.beta = 1
        self.classes = 2

        # self.bce = nn.BCEWithLogitsLoss()
        # self.CEL = nn.CrossEntropyLoss()
        # self.FL = Focal_Loss(weight=1,gamma=2)
        self.dice = DiceLoss(classes=2)
    ####todo 单通道




    def forward(self, input, target):
        target_expanded_BCE = expand_as_one_hot(target.long(), self.classes)
        # target_expanded = expand_as_one_hot(target.long(), self.classes)
        # target_expanded_BCE = target.unsqueeze(1)
        # print(input.size() )
        # print(target_expanded_BCE.size())
        assert input.size() == target_expanded_BCE.size(), "'input' and 'target' must have the same shape"
        devices = input.device

        input.to(devices)
        target_expanded_BCE.to(devices)
        eposion = 1e-10
        # #TODO 分通道
        # i0=input[:,0,:,:,:]
        # i1=input[:,1,:,:,:]
        # l0=target_expanded_BCE[:,0,:,:,:]
        # l1=target_expanded_BCE[:,1,:,:,:]
        #
        # fp = open(r'D:\Work\ZoomNet/loss_log.txt', "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
        # # sigmoid_pred = torch.sigmoid(input)
        # count_pos = torch.sum(l1) * 1.0 + eposion#计算一个batch有多少正样本
        # count_neg = torch.sum(1. - l1) * 1.0#计算一个batch有多少负样本
        # beta = count_neg / count_pos###计算负样本是正样本的倍数,tensor类型
        # beta_back = count_pos / (count_pos + count_neg)
        # # print("\n======== Beta is :",beta)
        # # bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)####随每次的batch变化而变化
        # #TODO 双通道
        # ## output = torch.full([10, 64], 0.999)  # A prediction (logit)
        # ## pos_weight = torch.ones([2])
        # # bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10,beta]).to(device))####随每次的batch变化而变化
        # bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([beta]).to(device))  ####随每次的batch变化而变化
        # bce1_0 = nn.BCEWithLogitsLoss()  ####随每次的batch变化而变化
        # bce1_1 = nn.BCEWithLogitsLoss()  ####随每次的batch变化而变化
        # bce1_1_weight = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([beta]).to(device))  ####随每次的batch变化而变化
        #
        # loss_11 = bce1(input, target_expanded_BCE)#2个通道同时输出的加权重的loss
        # loss_1_0 = bce1_0(i0, l0)#通道0=标签为0的loss
        # loss_1_1 = bce1_1(i1,l1)##通道1=标签为1的加权重的loss
        # loss_1_1weight= bce1_1_weight(i1,l1)#通道1=标签为1的加权重的loss
        #
        # loss_1=(loss_1_0+loss_1_1weight)/2###输出最终loss
        # print("\n======== Beta is :",beta,
        #     "\nloss_11 is ===> ", loss_11, "\nloss_1_0 is  ===>", loss_1_0, "\nloss_1_1 is ===>", loss_1_1, "\nloss_1_weight is ===>",
        #       loss_1_1weight, "\nloss_ is ", loss_1 ,"=======================\n",file=fp)
        # fp.close()

        #TODO 单通道
        ## bce1 = nn.BCEWithLogitsLoss()####随每次的batch变化而变化
        # bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([beta]).to(device))####随每次的batch变化而变化


        # print(input.is_cuda, target_expanded.is_cuda)
        # loss_1 = beta_back* bce1(input, target_expanded_BCE)
        # i0=input[:,0,:,:,:]
        # i1=input[:,1,:,:,:]
        # l0=target_expanded_BCE[:,0,:,:,:]
        #计算正负样本比例
        l1=target_expanded_BCE[:,1,:,:,:]
        # sigmoid_pred = torch.sigmoid(input)
        count_pos = torch.sum(l1) * 1.0 + eposion#计算一个batch有多少正样本
        count_neg = torch.sum(1. - l1) * 1.0#计算一个batch有多少负样本
        beta = count_neg / count_pos###计算负样本是正样本的倍数,tensor类型
        weights = [1.0, beta]
        # class_weights = torch.FloatTensor(weights).to(device)
        #交叉熵
        # CEL = nn.CrossEntropyLoss(weight=class_weights)
        # loss_1 = CEL(input, target_expanded_BCE)

        #focal loss

        # loss_1=self.FL(input, target)
        # input=nn.sigmoid(input)
        # dice_loss=dice_loss_1(input, target)
        # bce_loss=
        # ce_target=target.type(torch.LongTensor).to(device)
        loss_2, channel_score = self.beta * self.dice(input, target)
        # ce_loss=self.CEL(input,ce_target)
        return  (loss_2) , channel_score
        #
        # return (loss_1 )
class DiceLoss_ch_1(nn.Module):
    """Linear combination of BCE and Dice losses3D"""

    def __init__(self, alpha=1, beta=1, classes=4):
        super(DiceLoss_ch_1, self).__init__()
        # self.alpha = alpha#####平衡因子，用于平衡正负比例
        # self.bce = nn.BCEWithLogitsLoss()
        # self.beta = beta
        # self.dice = DiceLoss(classes=classes)
        # self.classes=classes
        self.alpha = 1  #####平衡因子，用于平衡正负比例
        self.beta = 1
        self.classes = 2

        self.bce = nn.BCEWithLogitsLoss()
        self.CEL = nn.CrossEntropyLoss()
        self.FL = Focal_Loss(weight=1,gamma=2)
        self.dice = DiceLoss(classes=2)
    ####todo 单通道




    def forward(self, input, target):
        target_expanded_BCE = expand_as_one_hot(target.long(), self.classes)
        # target_expanded = expand_as_one_hot(target.long(), self.classes)
        # target_expanded_BCE = target.unsqueeze(1)
        # print(input.size() )
        # print(target_expanded_BCE.size())
        assert input.size() == target_expanded_BCE.size(), "'input' and 'target' must have the same shape"
        device = torch.device('cuda:0')

        input.to(device)
        target_expanded_BCE.to(device)
        eposion = 1e-10
        # #TODO 分通道
        # i0=input[:,0,:,:,:]
        # i1=input[:,1,:,:,:]
        # l0=target_expanded_BCE[:,0,:,:,:]
        # l1=target_expanded_BCE[:,1,:,:,:]
        #
        # fp = open(r'D:\Work\ZoomNet/loss_log.txt', "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
        # # sigmoid_pred = torch.sigmoid(input)
        # count_pos = torch.sum(l1) * 1.0 + eposion#计算一个batch有多少正样本
        # count_neg = torch.sum(1. - l1) * 1.0#计算一个batch有多少负样本
        # beta = count_neg / count_pos###计算负样本是正样本的倍数,tensor类型
        # beta_back = count_pos / (count_pos + count_neg)
        # # print("\n======== Beta is :",beta)
        # # bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)####随每次的batch变化而变化
        # #TODO 双通道
        # ## output = torch.full([10, 64], 0.999)  # A prediction (logit)
        # ## pos_weight = torch.ones([2])
        # # bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10,beta]).to(device))####随每次的batch变化而变化
        # bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([beta]).to(device))  ####随每次的batch变化而变化
        # bce1_0 = nn.BCEWithLogitsLoss()  ####随每次的batch变化而变化
        # bce1_1 = nn.BCEWithLogitsLoss()  ####随每次的batch变化而变化
        # bce1_1_weight = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([beta]).to(device))  ####随每次的batch变化而变化
        #
        # loss_11 = bce1(input, target_expanded_BCE)#2个通道同时输出的加权重的loss
        # loss_1_0 = bce1_0(i0, l0)#通道0=标签为0的loss
        # loss_1_1 = bce1_1(i1,l1)##通道1=标签为1的加权重的loss
        # loss_1_1weight= bce1_1_weight(i1,l1)#通道1=标签为1的加权重的loss
        #
        # loss_1=(loss_1_0+loss_1_1weight)/2###输出最终loss
        # print("\n======== Beta is :",beta,
        #     "\nloss_11 is ===> ", loss_11, "\nloss_1_0 is  ===>", loss_1_0, "\nloss_1_1 is ===>", loss_1_1, "\nloss_1_weight is ===>",
        #       loss_1_1weight, "\nloss_ is ", loss_1 ,"=======================\n",file=fp)
        # fp.close()

        #TODO 单通道
        ## bce1 = nn.BCEWithLogitsLoss()####随每次的batch变化而变化
        # bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([beta]).to(device))####随每次的batch变化而变化


        # print(input.is_cuda, target_expanded.is_cuda)
        # loss_1 = beta_back* bce1(input, target_expanded_BCE)
        # i0=input[:,0,:,:,:]
        # i1=input[:,1,:,:,:]
        # l0=target_expanded_BCE[:,0,:,:,:]
        #计算正负样本比例
        l1=target_expanded_BCE[:,1,:,:,:]
        # sigmoid_pred = torch.sigmoid(input)
        count_pos = torch.sum(l1) * 1.0 + eposion#计算一个batch有多少正样本
        count_neg = torch.sum(1. - l1) * 1.0#计算一个batch有多少负样本
        beta = count_neg / count_pos###计算负样本是正样本的倍数,tensor类型
        weights = [1.0, beta]
        class_weights = torch.FloatTensor(weights).to(device)
        #交叉熵
        CEL = nn.CrossEntropyLoss(weight=class_weights)
        # loss_1 = CEL(input, target_expanded_BCE)

        #focal loss

        # loss_1=self.FL(input, target)
        # input=nn.sigmoid(input)
        # dice_loss=dice_loss_1(input, target)
        # bce_loss=

        loss_2, channel_score = self.beta * self.dice(input, target)

        return  (loss_2) , channel_score
        #
        # return (loss_1 )


