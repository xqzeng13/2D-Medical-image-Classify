import numpy as np
import torch
from torch import nn as nn
from torch.optim import lr_scheduler
from torch.nn.modules.loss import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from medpy import metric

import pandas as pd
import SimpleITK as sitk
import os
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter
from lib.losses3D.basic import expand_as_one_hot, expand_target
from lib.Evaluate.pre_process import data_transform
from lib.Evaluate.ROC_AUC import iou_score, precision
from lib.loss_function.focal_loss import BCEloss
from lib.loss_function.focal_loss import BCEloss_V2
from lib.loss_function.focal_loss import IOUloss

from sklearn.metrics import roc_curve, RocCurveDisplay, auc, accuracy_score

from lib.losses3D import DiceLoss
from tensorboardX import SummaryWriter
import logging


def calculate_metric_percase(pred, gt):
    # dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    # acc=metric.binary.
    # hd=0
    # asd=0
    # asd = metric.binary.asd(pred, gt)
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)
    # spe = metric.binary.specificity(pred, gt)
    # sensti = metric.binary.sensitivity(pred, gt)
    # miou=metric.
    return precision, recall

class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model,device,trainformat, criterion, optimizer, train_data_loader,
                 valid_data_loader=None,):

        self.args = args
        self.model = model
        self.save_csvname = trainformat
        self.criterion = criterion
        # self.criterion2=criterion_2
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        # self.len_epoch = 500
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None


        self.optimizer = optimizer
        self.lr_scheduler = None
        self.device=device
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
        self.lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 5
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1

        self.output_final = []
        self.target_final = []
        self.output_list = []
        self.target_list = []

        self.iouscore_list = []
        self.prescore_list = []
        self.epoch_list = []

    def training(self):
        output_list = []
        target_list = []
        early_stop = 15
        trigger = 0  ###计数器
        best = [0, np.inf]
        best_dice = 0.0001
        self.iter_num=0
        # device_ids = [3,4, 5, 6, 7]
        for epoch in range(self.start_epoch, self.args.nEpochs):
            start_time=time.time()
            self.iou_score_list = []
            self.pre_score_list = []
            self.output_final = []
            self.target_final = []
            self.output_list = []
            self.target_list = []
            trigger += 1
            # self.trainformat=
            # self.train_epoch(epoch)
            self.new_train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)
            self.val_score_log(epoch)

            # ===================ROC curve====================================
            # input, target1 = data_transform(self.output_final, self.target_final)
            # fpr, tpr, thresholds = roc_curve(target1[1], input[1])
            # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            # plt.title('ROC curve  Epoch = ' + str(epoch))
            # plt.show()
            # =======================================================
            # val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']
            # print(val_loss)
            val_dice = self.writer.data['val']['dsc'] / self.writer.data['val']['count']
            # 保存最佳model
            # if isinstance(self.model, nn.DataParallel):
            #     self.model1 = self.model.module
            #     self.model1=self.model1.to('cuda:1')

            if self.args.save is not None and ((epoch + 1) % self.save_frequency):  #########每3个epoch次 保存一个model
                if best_dice < val_dice:
                    best_dice = val_dice
                    # best_model=self.model.module.state_dict()
                    best_model=self.model.state_dict()

                    name = "{}_BEST.pth".format(os.path.basename(self.args.save))
                    # print(self.args.save)
                    torch.save(best_model, os.path.join(self.args.save, name))
            ###############保存最佳model并保存训练log############
            # if self.args.save is not None and ((epoch + 1) % self.save_frequency):  #########每3个epoch次 保存一个model
            #     # 如果是 nn.DataParallel 模型，则转换为 nn.Module
            #     if isinstance(self.model, nn.DataParallel):
            #         self.model = self.model.module
            #     self.model.save_checkpoint(self.args.save,
            #                                epoch, val_dice,
            #                                optimizer=self.optimizer)
            #
            #
            # self.model=self.model.cpu()
            # self.model = nn.DataParallel(self.model)
            #  # 模型导入GPU
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # self.model.to(device)
            # self.model.to(device_ids[0])
            #
            # torch.cuda.set_device(device_ids[0])
            # self.model = nn.DataParallel(self.model, device_ids=device_ids)
            # # print(torch.cuda.device_count())
            # self.model.to(device_ids[0])
            ############################
            # device_ids = [3,4, 5, 6, 7]#self.conv3d_c1_1(x)
            # if torch.cuda.device_count() > 1:
            #     torch.cuda.set_device(device_ids[0])
            #     self.model = nn.DataParallel(self.model, device_ids=device_ids)
            #     # print(torch.cuda.device_count())
            # self.model.to(device_ids[0])
            #####################
            self.writer.write_end_of_epoch(epoch)
            self.writer.reset('train')
            self.writer.reset('val')
            # self.lr_scheduler.step()
            ##TODO early stopping
            if val_loss<best[1]:
                best[0]=epoch
                best[1]=val_loss
                trigger=0
            end_time = time.time()
            epoch_time = end_time - start_time
            print("Epoch {} took {:.2f} seconds".format(epoch , epoch_time))
            if trigger>=early_stop:

                print("Early stopping!!!!!!!!!!!!!!")
                print("Epoch is :",best[0])
                break
    def new_train_epoch(self, epoch):

        self.model.train()

        # train_loss = LossAverage()
        # train_dice = DiceAverage(2)
        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()

            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)  #######(8,16,16,16)
            input_tensor=input_tensor.to(self.device)
            target=target.to(self.device)
            input_tensor.requires_grad = True

            # input_tensor = torch.randn(64, 1, 32, 32, 32)
            output= self.model(input_tensor)  ##输出单通道（64，1，16，16，16）log_dir


            loss_dice, per_ch_score = self.criterion(output, target)  ##把loss_dice,per_ch_scorec存入criterion中

            loss_dice.backward()  ##反向传播
            self.optimizer.step()  #########梯度下降
            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:  #################每100次展示一次loss一次处理batchsize批次数据
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

##调整学习率
        lr_ = self.args.lr * (1.0 - self.iter_num /self.args.nEpochs) ** 0.9
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch, ):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            self.output_list = []
            self.target_list = []

            with torch.no_grad():  #########反向传播时就不会自动求导了，因此大大节约了显存或者说内存
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False
                output= self.model(input_tensor)

                # output = self.model(input_tensor,edge)
                # 作为roc计算数据来源
                output1 = output  # (64,2,16,16,16)
                target1 = target  # (64,16,16,16)

                # TODO 双通道
                # output1=output1[:, 1, :, :, :]
                softmax_1 = nn.Softmax(dim=1)
                output1 = softmax_1(output1)
                # output1 = torch.argmax(output1, dim=1) #（64，16，16，16） ##返回dim维度上张量最大值的索引
                iou = iou_score(output1, target1)
                preci = precision(output1, target1)
                self.iou_score_list.append(iou)
                self.pre_score_list.append(preci)

                self.output_list.append(output1)
                self.target_list.append(target1)
                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

        # 将每个epoch的值传出
        self.output_final = self.output_list
        self.target_final = self.target_list

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)

    def val_score_log(self, epoch):

        IOU_Score = sum(self.iou_score_list)
        Precison = sum(self.pre_score_list)
        IOU_Score_ave = IOU_Score / len(self.iou_score_list)
        Pre_Score_ave = Precison / len(self.pre_score_list)
        self.iouscore_list.append(IOU_Score_ave)
        self.prescore_list.append(Pre_Score_ave)
        self.epoch_list.append(epoch)
        output_excel = {'epoch': [], 'Pre_Score_ave': [], 'IOU_Score_ave': []}
        output_excel['epoch'] = self.epoch_list
        output_excel['Pre_Score_ave'] = self.prescore_list
        output_excel['IOU_Score_ave'] = self.iouscore_list
        # output_excel['classification'] = classificationlist
        output_csv = pd.DataFrame(output_excel)
        csv_path=self.args.log_dir
        output_csv.to_csv(csv_path+self.save_csvname+'-val_curve_log.csv',
                          index=False)
        print("\n IOU_Score_ave  is : ", IOU_Score_ave, "\n Pre_Score_ave  is : ", Pre_Score_ave)
class Trainer_classify:
    """
    Trainer class
    """

    def __init__(self, args, model,device, trainformat,criterion, optimizer, train_data_loader,
                 valid_data_loader=None,):

        self.args = args
        self.model = model
        self.save_csvname = trainformat
        self.criterion = criterion
        # self.criterion2=criterion_2
        self.train_data_loader = train_data_loader
        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        # self.len_epoch = 500
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None


        self.optimizer = optimizer
        self.lr_scheduler = None
        self.device=device
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
        self.lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 5
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1
        self.writer = SummaryWriter(self.args.save + '/log')

        self.output_final = []
        self.target_final = []
        self.output_list = []
        self.target_list = []

        self.iouscore_list = []
        self.prescore_list = []
        self.epoch_list = []
        self.ce_loss = CrossEntropyLoss()

        # self.calculate_metric_percase=calculate_metric_percase()
    def training(self):
        output_list = []
        target_list = []
        early_stop = 30
        trigger = 0  ###计数器
        best = [0, np.inf]
        best_dice = 0.0001
        self.iter_num =0
        self.acc_max=0
        # device_ids = [3,4, 5, 6, 7]
        logging.info("{} iterations per epoch".format( self.len_epoch))

        for epoch in range(self.start_epoch, self.args.nEpochs):
            start_time=time.time()
            self.iou_score_list = []
            self.pre_score_list = []
            self.output_final = []
            self.target_final = []
            self.output_list = []
            self.target_list = []
            trigger += 1
            # self.trainformat=
            # self.train_epoch(epoch)
            self.new_train_epoch(epoch)
            self.model1_loss_sum=0
            # if self.do_validation:
            if self.iter_num%30 ==0:
                self.validate_epoch(epoch)
            end_time = time.time()
            epoch_time = end_time - start_time
            print("Epoch {} took {:.2f} seconds".format(epoch , epoch_time))

            # self.val_score_log(epoch)

            # ===================ROC curve====================================
            # input, target1 = data_transform(self.output_final, self.target_final)
            # fpr, tpr, thresholds = roc_curve(target1[1], input[1])
            # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            # plt.title('ROC curve  Epoch = ' + str(epoch))
            # plt.show()
            # =======================================================
            # val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']

            # 保存最佳model
            # if isinstance(self.model, nn.DataParallel):
            #     self.model1 = self.model.module
            #     self.model1=self.model1.to('cuda:1')
            #
            # if self.args.save is not None and ((epoch + 1) % self.save_frequency):  #########每3个epoch次 保存一个model
            #     if best_dice < val_dice:
            #         best_dice = val_dice
            #         # best_model=self.model.module.state_dict()
            #         best_model=self.model.state_dict()
            #
            #         name = "{}_BEST.pth".format(os.path.basename(self.args.save))
            #         # print(self.args.save)
            #         torch.save(best_model, os.path.join(self.args.save, name))
            #
            # # self.writer.write_end_of_epoch(epoch)
            # # self.writer.reset('train')
            # # self.writer.reset('val')
            # # self.lr_scheduler.step()
            # ##TODO early stopping
            # if val_loss<best[1]:
            #     best[0]=epoch
            #     best[1]=val_loss
            #     trigger=0
            # end_time = time.time()
            # epoch_time = end_time - start_time
            # print("Epoch {} took {:.2f} seconds".format(epoch , epoch_time))
            # if trigger>=early_stop:
            #
            #     print("Early stopping!!!!!!!!!!!!!!")
            #     print("Epoch is :",best[0])
            #     break
    def new_train_epoch(self, epoch):

        self.model.train()

        for batch_idx, input_tuple in enumerate(self.train_data_loader):
            self.optimizer.zero_grad()

            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)  #######(8,16,16,16)
            input_tensor=input_tensor.to(self.device)
            target=target.to(self.device)
            input_tensor.requires_grad = True

            output= self.model(input_tensor)  ##输出单通道（64，1，16，16，16）

            loss_ce=self.ce_loss(output, target)
            self.iter_num  = self.iter_num + 1
            loss_ce.backward()  ##反向传播
            self.optimizer.step()  #########梯度下降
            loss_ce.item()

            self.writer.add_scalar('loss/model_loss',
                          loss_ce, self.iter_num )

            print('iteration %d : model1 loss : %f ' % (self.iter_num , loss_ce.item()))
            logging.info(
            'iteration %d : model1 loss : %f ' % (self.iter_num , loss_ce.item()))
##调整学习率
        lr_ = self.args.lr * (1.0 - self.iter_num /self.args.nEpochs) ** 0.9
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_



    def validate_epoch(self, epoch, ):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            self.output_list = []
            self.target_list = []

            with torch.no_grad():  #########反向传播时就不会自动求导了，因此大大节约了显存或者说内存
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False
                output= self.model(input_tensor)

                # output = self.model(input_tensor,edge)
                # 作为roc计算数据来源
                output1 = output  # (64,2,16,16,16)
                target1 = target  # (64,16,16,16)

                # TODO 双通道
                # output1=output1[:, 1, :, :, :]
                softmax_1 = nn.Softmax(dim=1)
                output1 = softmax_1(output1)
                output_pse=torch.argmax(output1,dim=1)
                # output1 = torch.argmax(output1, dim=1) #（64，16，16，16） ##返回dim维度上张量最大值的索引
                precision, recall=calculate_metric_percase(np.array(output_pse.detach().cpu()),np.array(target1.detach().cpu()))
                pred_flatt = output_pse.ravel().detach().cpu()
                gt_flatt = target1.ravel().detach().cpu()
                acc = accuracy_score(gt_flatt, pred_flatt)
                f1=2*(precision*recall)/(precision+recall+0.000001)
                if f1>self.acc_max:
                    self.acc_max=f1
                    best_model=self.model.state_dict()
                #
                    name = "{}_BEST.pth".format(os.path.basename(self.args.save))
                #         # print(self.args.save)
                    torch.save(best_model, os.path.join(self.args.save, name))
                print('iteration %d :  ' % (self.iter_num ),"f1=",f1,"acc=",acc,"pre=",precision,'recall=',recall)
                # iou = iou_score(output1, target1)
                # preci = precision(output1, target1)
                # self.iou_score_list.append(iou)
                # self.pre_score_list.append(preci)
                #
                # self.output_list.append(output1)
                # self.target_list.append(target1)


    def val_score_log(self, epoch):

        IOU_Score = sum(self.iou_score_list)
        Precison = sum(self.pre_score_list)
        IOU_Score_ave = IOU_Score / len(self.iou_score_list)
        Pre_Score_ave = Precison / len(self.pre_score_list)
        self.iouscore_list.append(IOU_Score_ave)
        self.prescore_list.append(Pre_Score_ave)
        self.epoch_list.append(epoch)
        output_excel = {'epoch': [], 'Pre_Score_ave': [], 'IOU_Score_ave': []}
        output_excel['epoch'] = self.epoch_list
        output_excel['Pre_Score_ave'] = self.prescore_list
        output_excel['IOU_Score_ave'] = self.iouscore_list
        # output_excel['classification'] = classificationlist
        output_csv = pd.DataFrame(output_excel)
        csv_path=r'/home/lisic/zengxq/MedicalZoo_for_test/runs/IOU_PRE_curve/'
        output_csv.to_csv(csv_path+self.save_csvname+'-val_curve_log.csv',
                          index=False)
        print("\n IOU_Score_ave  is : ", IOU_Score_ave, "\n Pre_Score_ave  is : ", Pre_Score_ave)


    # %%
