#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python 3.7 plz



# blue : go
# red  : debugs
# green : import
import crayons
import pickle
import pandas as pd
import os , sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))  # 1階層上のfileのimport
import configuration

from krg_cgp_cnn.cgp import *
from krg_cgp_cnn.cgp_config import *
from krg_cgp_cnn.cnn_train import CNN_train

def makefiles():
    for trial in range(cnf.num_trial):
        os.makedirs(cnf.res_path+"/trial_{}".format(trial),exist_ok=True)


if __name__ == '__main__':

    cnf = configuration.Configuration()
    cnf.setting()
    os.makedirs(cnf.res_path,exist_ok=True)
    makefiles()

    for trial in range(cnf.num_trial):  
        print(crayons.blue("### Reset seed and trial ", trial ,"###"))
        cnf.torch_fix_seed(trial)
        cnf.trial_path = cnf.res_path + "/trial_{}".format(trial)
        # --- Optimization of the CNN architecture ---
        if cnf.CGP_mode == 'evolution':
            # Create CGP configuration and save network information
            # In CGP Config
            cnf.set_CGP()
            network_info = CgpInfoConvSet(rows=cnf.rows, cols=cnf.cols, level_back=cnf.level_back, min_active_num=cnf.min_active_num, max_active_num=cnf.max_active_num)
            with open(os.path.join(cnf.trial_path, 'network_info.pkl'), mode='wb') as f:
                pickle.dump(network_info, f)
            # Evaluation function for CGP (training CNN and return validation accuracy)
            imgSize = 32
            cnf.set_CNN()
            eval_f = CNNEvaluation(cnf,gpu_num=cnf.gpu_num, dataset='cifar10', reduced=cnf.reduced, verbose=True, epoch_num=cnf.epoch_num, 
                                batchsize=128, imgSize=imgSize)  # In cgp_config
            # Execute evolution
            cnf.set_G2V()
            cgp = CGP(network_info, eval_f, cnf, lam=cnf.lam, imgSize=imgSize, init=cnf.init, bias=cnf.bias,G2V_pop=cnf.G2V_pop)
            # Graph2Vec
            cgp.G2V_initialization()
            # RBF SAEA
            cgp._evaluation_val()
            # CGP-CNN
            cgp.modified_evolution(max_eval=cnf.max_eval, mutation_rate=cnf.mutation_rate, log_path=cnf.trial_path)

    if cnf.sub_mode == 'analytics':
        print(crayons.red("Analytics"))


    # --- Retraining evolved architecture ---
    if cnf.mode == 'retrain':
        print('Retrain')
        # In the case of existing log_cgp.txt
        # Load CGP configuration
        with open(os.path.join(cnf.log_path, 'network_info.pkl'), mode='rb') as f:
            network_info = pickle.load(f)
        # Load network architecture
        cgp = CGP(network_info, None)
        data = pd.read_csv(os.path.join(cnf.log_path, 'log_cgp.txt'), header=None)  # Load log file
        cgp.load_log(list(data.iloc[[cnf.epoch_load - 1]].values.flatten().astype(float)))  # Read the log at final generation
        print(cgp._log_data(net_info_type='active_only', start_time=0))
        # Retraining the network
        temp = CNN_train('cifar10', reduced=cnf.reduced, validation=False, verbose=True, batchsize=128)
        acc, macs = temp(cgp.pop[0].active_net_list(), 0, epoch_num=500, out_model='retrained_net.model')
        print(acc, macs)

        # # otherwise (in the case where we do not have a log file.)
        # temp = CNN_train('haze1', validation=False, verbose=True, imgSize=128, batchsize=16)
        # cgp = [['input', 0], ['S_SumConvBlock_64_3', 0], ['S_ConvBlock_64_5', 1], ['S_SumConvBlock_128_1', 2], ['S_SumConvBlock_64_1', 3], ['S_SumConvBlock_64_5', 4], ['S_DeConvBlock_3_3', 5]]
        # acc = temp(cgp, 0, epoch_num=500, out_model='retrained_net.model')

    elif cnf.mode == 'reevolution':
        # restart evolution
        print('Restart Evolution')
        imgSize = 32
        with open(os.path.join(cnf.log_path, 'network_info.pkl'), mode='rb') as f:
            network_info = pickle.load(f)
        eval_f = CNNEvaluation(gpu_num=cnf.gpu_num, dataset='cifar10', reduced=cnf.reduced, verbose=True, epoch_num=50, batchsize=128,
                               imgSize=imgSize)
        cgp = CGP(network_info, eval_f, lam=cnf.lam, imgSize=imgSize, bias=cnf.bias)

        data = pd.read_csv(os.path.join(cnf.log_path, 'log_cgp.txt'), header=None)
        cgp.load_log(list(data.tail(1).values.flatten().astype(float)))
        cgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_path=cnf.log_path)

    else:
        print('Undefined mode. Please check the "-m evolution or retrain or reevolution" ')
