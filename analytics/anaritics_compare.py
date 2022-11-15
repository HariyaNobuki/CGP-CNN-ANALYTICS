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

def summary_CGP():
    print("---",trial)
    generation_list = []
    num_eval_list = []
    evaluation_list = []
    time_list = []
    size_list = []
    active_node_list = []

    read_df = pd.read_csv(res_path + "/_log_cgp.csv")
    for i in range(cnf.max_eval):
        obj_list = read_df.query("generation=={}".format(i))
        min_idx = obj_list["evaluation"].idxmin()
        generation_list.append(i)
        num_eval_list.append(obj_list["num_eval"][min_idx])
        evaluation_list.append(obj_list["evaluation"][min_idx])
        time_list.append(obj_list["time"][min_idx])
        size_list.append(obj_list["size"][min_idx])
        active_node_list.append(obj_list["active_node"][min_idx])

    df_CGP = pd.DataFrame(
        {
            "generation".format(trial):generation_list,
            "num_eval".format(trial):num_eval_list,
            "time_{}".format(trial):time_list,
            "evaluation_{}".format(trial):evaluation_list,
            "size_{}".format(trial):size_list,
            "active_node_{}".format(trial):active_node_list,
        }
    )
    if os.path.isfile(mode_path + "/_log_cgp_summary.csv"):
        df_o = pd.read_csv(mode_path + "/_log_cgp_summary.csv")
        df_m = pd.concat([df_o,df_CGP],axis=1)        # merge
        df_m.to_csv(mode_path + "/_log_cgp_summary.csv",index = False)
    else:
        df_CGP.to_csv(mode_path + "/_log_cgp_summary.csv",index = False)

def stats_CGP():
    print("---stats")
    cgp_list = ["evaluation"]
    df_stats_CGP = pd.read_csv(mode_path + "/_log_cgp_summary.csv")
    for point in cgp_list:
        df_query = df_stats_CGP.filter(like=point,axis=1)   # 条件抽出
        df_quantile = pd.DataFrame({
        "eval"  : df_stats_CGP["num_eval"],
        "q1"    : np.quantile(df_query , 0     , axis = 1),
        "q2"    : np.quantile(df_query , 0.25  , axis = 1),
        "q3"    : np.quantile(df_query , 0.5   , axis = 1),
        "q4"    : np.quantile(df_query , 0.75  , axis = 1),
        "q5"    : np.quantile(df_query , 1.0   , axis = 1),
        })
        df_quantile.to_csv(mode_path + "/_log_cgp_{}.csv".format(point),index = False)

def make_graph_CGP(df):
    plt.plot(df["eval"],df["q1"],marker=".",ms=3,lw=1,label=mode)

def summary_CNN():
    print("---",trial)
    train_loss_list = []
    train_acc_list = []
    time_list = []
    test_loss_list = []
    test_acc_list = []

    read_df = pd.read_csv(res_path + "/_log_cnn.csv")

    obj_list = read_df.query("epoch==50")

    df_CNN = pd.DataFrame(
        {
            "eval"                        :[i+1 for i in range(len(obj_list))],
            "train_loss_{}".format(trial) :obj_list["train_loss"],
            "train_acc_{}".format(trial)  :obj_list["train_acc"],
            "time_list_{}".format(trial)  :obj_list["time"],
            "test_loss_{}".format(trial)  :obj_list["test_loss"],
            "test_acc_{}".format(trial)   :obj_list["test_acc"],
        }
    )
    df_CNN = df_CNN.reset_index(drop=True)
    if os.path.isfile(mode_path + "/_log_cnn_summary.csv"):
        df_o = pd.read_csv(mode_path + "/_log_cnn_summary.csv")
        df_m = pd.concat([df_o,df_CNN],axis=1)        # merge
        df_m.to_csv(mode_path + "/_log_cnn_summary.csv",index = False)
    else:
        df_CNN.to_csv(mode_path + "/_log_cnn_summary.csv",index = False)

def stats_CNN():
    print("---stats")
    cnn_list = ["train_loss","train_acc","test_loss","test_acc"]
    df_stats_CNN = pd.read_csv(mode_path + "/_log_cnn_summary.csv")
    for point in cnn_list:
        df_query = df_stats_CNN.filter(like=point,axis=1)   # 条件抽出
        df_quantile = pd.DataFrame({
        "eval"  : df_stats_CNN["eval"],
        "q1"    : np.quantile(df_query , 0     , axis = 1),
        "q2"    : np.quantile(df_query , 0.25  , axis = 1),
        "q3"    : np.quantile(df_query , 0.5   , axis = 1),
        "q4"    : np.quantile(df_query , 0.75  , axis = 1),
        "q5"    : np.quantile(df_query , 1.0   , axis = 1),
        })
        df_quantile.to_csv(mode_path + "/_log_cnn_{}.csv".format(point),index = False)

def make_graph_CNN(df):
    plt.plot(df["eval"],df["q1"],marker=".",ms=3,lw=1,label=mode)


if __name__ == '__main__':
    print("### CNN ANALYTICS ###")
    cnf = configuration.Configuration()
    cnf.setting()
    cnf.set_analitics()
    result_list = ["hry_d_10","hry_d_20","sgnm"]
    num_trial = [2,2,2]
    main_path = os.getcwd()
    sum_path = main_path + "/result_summary"

    # loss編
    if cnf.ANA_CGP == True:
        """分析段階"""
        mode_count = 0
        for mode in result_list:
            mode_path = sum_path + "/" + mode
            print("### ",mode," ###")
            for trial in range(num_trial[mode_count]):
                res_path = sum_path+"/"+mode+"/result/trial_"+str(trial)
                summary_CGP()
                stats_CGP()
            mode_count += 1
        """描画段階"""
        # res : sum_path
        fig_loss_eval = plt.figure(figsize=(8,5))
        for mode in result_list:
            mode_path = sum_path + "/" + mode
            df_eva = pd.read_csv(mode_path+"/_log_cgp_evaluation.csv")
            make_graph_CGP(df_eva)
        plt.legend()
        plt.xlim(0,60)
        plt.ylim(10,20)
        plt.xlabel("eval")
        plt.ylabel("loss")
        fig_loss_eval.savefig(sum_path+"/_summary_eval_fitness.png")

    # acc編
    if cnf.ANA_CNN == True:
        mode_count = 0
        for mode in result_list:
            mode_path = sum_path + "/" + mode
            print("### ",mode," ###")
            for trial in range(num_trial[mode_count]):
                res_path = sum_path+"/"+mode+"/result/trial_"+str(trial)
                summary_CNN()
            stats_CNN()
            mode_count += 1
        """描画段階"""
        # res : sum_path
        fig_list = ["train_loss","train_acc","test_loss","test_acc"]
        for fig_i in fig_list:
            print("### FIG ",fig_i," ###")
            fig = plt.figure(figsize=(8,5))
            for mode in result_list:
                mode_path = sum_path + "/" + mode
                df_eva = pd.read_csv(mode_path+"/_log_cnn_{}.csv".format(fig_i))
                make_graph_CGP(df_eva)
            plt.legend()
            #plt.xlim(0,60)
            #plt.ylim(10,20)
            plt.xlabel("eval")
            plt.ylabel("acc")
            fig.savefig(sum_path+"/_summary_cnn_{}.png".format(fig_i))

    #for trial in range(cnf.num_trial):
    #    trial_path = res_path + "/trial_{}".format(trial)
    #    df_best_CGP = analitics_CGP()
    #    make_graph_CGP(df_best_CGP)
    #    df_best_CNN = analitics_CNN()
    #    df_best_SAEA = analitics_SAEA()
