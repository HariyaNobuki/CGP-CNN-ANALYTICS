#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python 3.7 plz

# blue : g
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

sys.path.append(os.path.join(os.path.dirname(__file__), 'analytics'))  # 1階層上のfileのimport
import analytics_main

def makefiles():
    for trial in range(cnf.num_trial):
        os.makedirs(cnf.res_path+"/trial_{}".format(trial),exist_ok=True)


if __name__ == '__main__':

    cnf = configuration.Configuration()
    cnf.setting()
    os.makedirs(cnf.res_path,exist_ok=True)
    makefiles()

    if cnf.sub_mode == 'analytics':
        # single analytics
        print(crayons.red("Analytics"))
        cnf.set_ANA()
        ana_main = analytics_main.Analytics()
        cnf.date_path = cnf.res_path + "/" + cnf.res_date
        for exp in cnf.exp_mode:
            print(crayons.red("Mode ",bold=True),end="")
            i_path = cnf.date_path + "/" + exp
            for trial in range(cnf.num_trial):
                print('{}'.format(crayons.red('--trial')))
                df = pd.read_csv(i_path + "/trial_" + str(trial) + "/_log_cgp.csv")
                ana_main.analytics_log_cgp(df,i_path + "/trial_" + str(trial))
                df = pd.read_csv(i_path + "/trial_" + str(trial) + "/_log_epoch_test.csv")
                #ana_main.analytics_log_epoch_test(df,i_path + "/trial_" + str(trial))
                df = pd.read_csv(i_path + "/trial_" + str(trial) + "/_log_local_refinement_best.csv")
                ana_main.analytics_log_local_refinement_best(df,i_path + "/trial_" + str(trial))
                df = pd.read_csv(i_path + "/trial_" + str(trial) + "/_log_surrogate.csv")
                ana_main.analytics_log_surrogate(df,i_path + "/trial_" + str(trial))

