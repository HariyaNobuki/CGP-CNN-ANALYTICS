#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python 3.7 plz

import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os , sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))  # 1階層上のfileのimport
import configuration

def makefiles():
    for trial in range(cnf.num_trial):
        os.makedirs(cnf.res_path+"/trial_{}".format(trial),exist_ok=True)

def analitics_CGP():
    print("---CGP")
    generation_list = []
    num_eval_list = []
    evaluation_list = []
    time_list = []
    size_list = []
    active_node_list = []

    read_df = pd.read_csv(trial_path + "/_log_cgp.csv")
    for i in range(cnf.max_eval):
        obj_list = read_df.query("generation=={}".format(i))
        min_idx = obj_list["evaluation"].idxmin()
        generation_list.append(i)
        num_eval_list.append(obj_list["num_eval"][min_idx])
        evaluation_list.append(obj_list["evaluation"][min_idx])
        time_list.append(obj_list["time"][min_idx])
        size_list.append(obj_list["size"][min_idx])
        active_node_list.append(obj_list["active_node"][min_idx])

    df_best = pd.DataFrame(
        {
            "generation":generation_list,
            "num_eval":num_eval_list,
            "time":time_list,
            "evaluation":evaluation_list,
            "size":size_list,
            "active_node":active_node_list,
        }
    )
    df_best.to_csv(trial_path+"/_log_cgp_best.csv")
    return df_best

def make_graph_CGP(df_best):
    print("### Make Graph For CGP ###")
    fig_loss_eval = plt.figure()
    plt.plot(df_best["num_eval"],df_best["evaluation"])
    fig_loss_eval.savefig(trial_path+"/_fig_loss_eval.png")


#def make_graph_CGP(df_best):
#    print("### Make Graph ###")
#    fig=plt.figure()
#    ax1 = fig.add_subplot(2, 2, 1)
#    ax2 = fig.add_subplot(2, 2, 2)
#    ax3 = fig.add_subplot(2, 2, 3)
#    ax4 = fig.add_subplot(2, 2, 4)
#
#    ax1.plot(df_best["num_eval"],df_best["evaluation"])
#    ax1.set_xlabel("num_eval")
#    ax1.set_ylabel("CE loss")
#    ax2.plot(df_best["num_eval"],df_best["time"])
#    ax2.set_xlabel("num_eval")
#    ax2.set_ylabel("time")
#    ax3.plot(df_best["num_eval"],df_best["size"])
#    ax3.set_xlabel("num_eval")
#    ax3.set_ylabel("size")
#    ax4.plot(df_best["num_eval"],df_best["active_node"])
#    ax4.set_xlabel("num_eval")
#    ax4.set_ylabel("active_node")
#
#    fig.tight_layout()
#    fig.savefig(trial_path+"/summary.png")
#    plt.clf()
#    plt.close()
#
#def analitics_CNN():
#    print("---CNN")
#    generation_list = []
#    num_eval_list = []
#    evaluation_list = []
#    time_list = []
#    size_list = []
#    active_node_list = []
#
#    read_df = pd.read_csv(trial_path + "/_log_cgp.csv")
#    for i in range(cnf.max_eval):
#        obj_list = read_df.query("generation=={}".format(i))
#        min_idx = obj_list["evaluation"].idxmin()
#        generation_list.append(i)
#        num_eval_list.append(obj_list["num_eval"][min_idx])
#        evaluation_list.append(obj_list["evaluation"][min_idx])
#        time_list.append(obj_list["time"][min_idx])
#        size_list.append(obj_list["size"][min_idx])
#        active_node_list.append(obj_list["active_node"][min_idx])
#
#    df_best = pd.DataFrame(
#        {
#            "generation":generation_list,
#            "num_eval":num_eval_list,
#            "time":time_list,
#            "evaluation":evaluation_list,
#            "size":size_list,
#            "active_node":active_node_list,
#        }
#    )
#    df_best.to_csv(trial_path+"/_log_cgp_best.csv")
#    return df_best

def analitics_SAEA():
    print("---CNN")
    generation_list = []
    num_eval_list = []
    evaluation_list = []
    time_list = []
    kendalltau_list = []
    rank_dif_list = []
    rmse_list = []

    read_df = pd.read_csv(trial_path + "/_log_surrogate.csv")
    #for i in range(cnf.max_eval):
    for i in range(cnf.max_eval):
        i += 1
        obj_list = read_df.query("generation=={}".format(i))
        generation_list.append(i)
        num_eval_list.append(obj_list["num_eval"])
        time_list.append(obj_list["time"])
        kendalltau_list.append(obj_list["kendalltau"])
        rank_dif_list.append(obj_list["rank_dif"])
        rmse_list.append(obj_list["rmse"])

    df_SAEA = pd.DataFrame(
        {
            "generation":generation_list,
            "num_eval":num_eval_list,
            "time":time_list,
            "kendalltau":kendalltau_list,
            "rank_dif":rank_dif_list,
            "rmse":rmse_list,
        }
    )

    fig = plt.figure()
    plt.plot(num_eval_list , kendalltau_list)
    fig.savefig(trial_path+"/kendaltau.png")
    #df_SAEA.to_csv(trial_path+"/_log_cgp_best.csv")
    return df_SAEA


if __name__ == '__main__':
    print("### CNN ANALYTICS ###")
    cnf = configuration.Configuration()
    cnf.setting()
    main_path = os.getcwd()
    res_path = main_path + "/result"

    cnf.num_trial = 1

    for trial in range(cnf.num_trial):
        trial_path = res_path + "/trial_{}".format(trial)
        df_best_CGP = analitics_CGP()
        make_graph_CGP(df_best_CGP)
        df_best_CNN = analitics_CNN()
        df_best_SAEA = analitics_SAEA()
