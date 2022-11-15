#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import tqdm
import pandas as pd
import os , sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import init

from cnn_model import CGP2CNN
from my_data_loader import get_train_valid_loader
from datasets import CIFAR10Red


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 0.02, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self,cnf, dataset_name, reduced=False, validation=True, verbose=True, imgSize=32, batchsize=128):
        # dataset_name: name of data set ('bsds'(color) or 'bsds_gray')
        # validation: [True]  model train/validation mode
        #             [False] model test mode for final evaluation of the evolved model
        #                     (raining data : all training data, test data : all test data)
        # verbose: flag of display
        self.verbose = verbose
        self.imgSize = imgSize
        self.validation = validation
        self.batchsize = batchsize
        self.dataset_name = dataset_name
        self.cnf = cnf

        self.df_cnn_log = []

        # load dataset
        if dataset_name == 'cifar10':
            self.n_class = 10
            self.channel = 3
            if self.validation:
                # make validation
                print("### Make Validation ###")
                self.dataloader, self.test_dataloader = get_train_valid_loader(data_dir='./',
                                                                               batch_size=self.batchsize,
                                                                               augment=True, reduced=reduced,
                                                                               random_seed=2018, num_workers=1, 
                                                                               pin_memory=True)
                # self.dataloader, self.test_dataloader = loaders[0], loaders[1]
            else:
            
                if not reduced:
                    train_dataset = dset.CIFAR10(root='./', train=True, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.Scale(self.imgSize),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 ]))
                    test_dataset = dset.CIFAR10(root='./', train=False, download=True,
                                                transform=transforms.Compose([
                                                    transforms.Scale(self.imgSize),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                ]))
                else:
                    train_dataset = CIFAR10Red(root='./', train=True, download=True,
                                               transform=transforms.Compose([
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.Scale(self.imgSize),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                               ]))
                    test_dataset = CIFAR10Red(root='./', train=False, download=True,
                                              transform=transforms.Compose([
                                                  transforms.Scale(self.imgSize),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                              ]))
                
                self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize,
                                                              shuffle=True, num_workers=int(2))
                self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize,
                                                                   shuffle=True, num_workers=int(2))
            print('train num    ', len(self.dataloader.dataset))
            print('test num     ', len(self.test_dataloader.dataset))
        else:
            print('\tInvalid input dataset name at CNN_train()')
            exit(1)

    # evaluation
    # cgp -> 実質的にはnet
    def __call__(self, cgp, gpuID, epoch_num=50, out_model='mymodel.model'):   # df epoch_num=200
        if self.verbose:        # 認識すべき内容はこの程度
            #epoch_num=3         # ここの定義を頑張ろうか
            print('GPUID     :', gpuID)
            print('epoch_num :', epoch_num)
            print('batch_size:', self.batchsize)

        # model
        # 再現性をあげて速度が落ちる->True(これは毎回呼びださないといけない関数なのか？)
        torch.backends.cudnn.benchmark = True
        # self.channelの意味は不明
        model = CGP2CNN(cgp, self.channel, self.n_class, self.imgSize)  # ここが理解できるとtorchの本質的な部分に
        init_weights(model, 'kaiming')
        model.cuda(gpuID)
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        criterion.cuda(gpuID)
        optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, dampening=0, weight_decay=0.0005)
        # torch.Size([128, 3, 32, 32])
        input = torch.FloatTensor(self.batchsize, self.channel, self.imgSize, self.imgSize) # 32bitの浮動小数点
        input = input.cuda(gpuID)
        # torch.Size([128])
        label = torch.LongTensor(self.batchsize) # 64bitの浮動小数点
        label = label.cuda(gpuID)

        # Train loop    # こんなのはどう考えても時間がかかる

        for epoch in range(1, epoch_num + 1):
            start_time = time.time()
            if self.verbose:
                print('epoch', epoch)
            train_loss = 0
            total = 0
            correct = 0
            ite = 0
            for module in model.children():
                module.train(True)  # テスト中であることを通知するだけ
            for _, (data, target) in tqdm.tqdm(enumerate(self.dataloader)):    # batchごとに書いている
                data = data.cuda(gpuID)
                target = target.cuda(gpuID)
                input.resize_as_(data).copy_(data)
                input_ = Variable(input)
                label.resize_as_(target).copy_(target)
                label_ = Variable(label)
                optimizer.zero_grad()
                try:
                    output = model(input_, None)
                except:
                    import traceback
                    traceback.print_exc()
                    return 0.
                loss = criterion(output, label_)
                train_loss += loss.data.tolist()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(output.data, 1)
                total += label_.size(0)
                correct += predicted.eq(label_.data).cpu().sum()
                ite += 1
            print('Train set : Average loss: {:.4f}'.format(train_loss))
            print('Train set : Average Acc : {:.4f}'.format(correct / total))
            print('time ', time.time() - start_time)
            t_loss , t_acc= self.__test_per_std(model, criterion, gpuID, input, label)
            #self.df_cnn_log.append(self._log_data(epoch,train_loss,(correct/total),time.time()-start_time,t_loss,t_acc))

            if self.validation:
                if epoch == 30:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch == epoch_num:
                    for module in model.children():
                        module.train(False)
                    t_loss , t_acc= self.__test_per_std(model, criterion, gpuID, input, label)
                    
            else:
                if epoch == 5:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch % 10 == 0:
                    for module in model.children():
                        module.train(False)
                    t_loss , t_acc= self.__test_per_std(model, criterion, gpuID, input, label)
                    #self.df_cnn_log.append(self._log_data(epoch,train_loss,(correct / total),time.time()-start_time,t_loss,t_acc))
                if epoch == 250:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
                if epoch == 375:
                    for param_group in optimizer.param_groups:
                        tmp = param_group['lr']
                    tmp *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = tmp
        # save the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.save(model.state_dict(), './model_%d.pth' % int(gpuID))   # これはどこに保存されているの？
        return float(t_loss), num_params

    # For validation/test
    def __test_per_std(self, model, criterion, gpuID, input, label):
        test_loss = 0
        total = 0
        correct = 0
        ite = 0
        for _, (data, target) in enumerate(self.test_dataloader):
            data = data.cuda(gpuID)
            target = target.cuda(gpuID)
            input.resize_as_(data).copy_(data)
            input_ = Variable(input)
            label.resize_as_(target).copy_(target)
            label_ = Variable(label)
            try:
                output = model(input_, None)
            except:
                import traceback
                traceback.print_exc()
                return 0.
            loss = criterion(output, label_)
            test_loss += loss.data.tolist()
            _, predicted = torch.max(output.data, 1)
            total += label_.size(0)
            correct += predicted.eq(label_.data).cpu().sum()
            ite += 1
        print('Test set : Average loss: {:.4f}'.format(test_loss))
        print('Test set : (%d/%d)' % (correct, total))
        print('Test set : Average Acc : {:.4f}'.format(correct / total))

        return test_loss,(correct / total)

    def _log_data(self,epoch,train_loss,train_acc,time,test_loss,test_acc):
        log_list = {"epoch":epoch,"train_loss":train_loss,"train_acc":float(train_acc),
                    "time":time,"test_loss":test_loss, "test_acc":float(test_acc)}
        return log_list
    
    def _log_save(self):
        df_n = pd.DataFrame(self.df_cnn_log)        # new
        if os.path.isfile(self.cnf.trial_path + "/_log_cnn.csv"):
            df_o = pd.read_csv(self.cnf.trial_path + "/_log_cnn.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.trial_path + "/_log_cnn.csv",index = False)
        else:
            df_n.to_csv(self.cnf.trial_path + "/_log_cnn.csv",index = False)