#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import multiprocessing.pool
import numpy as np
import crayons
import krg_cgp_cnn.cnn_train as cnn


# wrapper function for multiprocessing
def arg_wrapper_mp(args):
    return args[0](*args[1:])


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


# Evaluation of CNNs
def cnn_eval(cnf, net, gpu_id, epoch_num, batchsize, dataset, reduced, verbose, imgSize):
    print('\tgpu_id:', gpu_id, ',', net)
    # コンストラクタ
    train = cnn.CNN_train(cnf , dataset, reduced=reduced, validation=True, verbose=verbose, imgSize=imgSize, batchsize=batchsize) # datasetの成型用関数的な役割を担っている
   # メソッド 
    evaluation = train(net, gpu_id, epoch_num=epoch_num, out_model=None)
    train._epoch_train_save()
    train._epoch_test_save()
    print(crayons.blue("eval\t:\t"), evaluation[0])
    print(crayons.blue("param\t:\t"), evaluation[1])
    return evaluation

# ここを無視するとデバッグ効率が壊滅的になる
# vervoseとかは自分で設定しているのか？
class CNNEvaluation(object):
    def __init__(self,cnf, gpu_num, dataset='cifar10', reduced=False, verbose=True, epoch_num=50, batchsize=16, imgSize=32):
        self.gpu_num = gpu_num
        #self.epoch_num = epoch_num
        self.batchsize = batchsize
        self.dataset = dataset
        self.reduced = reduced
        self.verbose = verbose
        self.imgSize = imgSize
        self.cnf = cnf

    def __call__(self, net_lists):
        evaluations = []
        for i in np.arange(0, len(net_lists), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(net_lists))) - i
            pool = NoDaemonProcessPool(process_num) # for multi process
            arg_data = [(cnn_eval,self.cnf, net_lists[i + j], j, self.cnf.epoch_num, self.batchsize, self.dataset, self.reduced, self.verbose,
                         self.imgSize) for j in range(process_num)] # そのままで挿入するだけ
            out = pool.map(arg_wrapper_mp, arg_data)
            for j in range(process_num):
                evaluations.append(out[j])
            pool.terminate()

        return evaluations



# network configurations
class CgpInfoConvSet(object):
    def __init__(self, rows=30, cols=40, level_back=40, min_active_num=8, max_active_num=50):
        self.input_num = 1
        # "S_" means that the layer has a convolution layer without downsampling.
        # "D_" means that the layer has a convolution layer with downsampling.
        # "Sum" means that the layer has a skip connection.
        # self.func_type = ['S_ConvBlock_32_1', 'S_ConvBlock_32_3', 'S_ConvBlock_32_5',
        #                   'S_ConvBlock_128_1', 'S_ConvBlock_128_3', 'S_ConvBlock_128_5',
        #                   'S_ConvBlock_64_1', 'S_ConvBlock_64_3', 'S_ConvBlock_64_5',
        #                   'S_ResBlock_32_1', 'S_ResBlock_32_3', 'S_ResBlock_32_5',
        #                   'S_ResBlock_128_1', 'S_ResBlock_128_3', 'S_ResBlock_128_5',
        #                   'S_ResBlock_64_1', 'S_ResBlock_64_3', 'S_ResBlock_64_5',
        #                   'Concat', 'Sum',
        #                   'Max_Pool', 'Avg_Pool'] # ここの作り方に依存する
        self.func_type = ['S_ConvBlock_32_3', 'S_ConvBlock_64_3', 'S_ConvBlock_128_3',
                          'S_InceptionResA_0_0', 'S_InceptionResB_0_0', 'S_InceptionResC_0_0', 
                          'D_InceptionResDiv1_0_0', 'D_InceptionResDiv2_0_0', 'Concat', 
                          'Sum', 'Max_Pool', 'Avg_Pool']   # 計12

        self.func_in_num = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 2, 
                            2, 1, 1]    # 計12

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])