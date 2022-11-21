import crayons  # print('{}'.format(crayons.red('red')))
import random
import os , sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Analytics:
    def __init__(self):
        print(crayons.red("### ANALYTICS ###"))

    def analytics_log_cgp(self,df,save_path):
        print('cgp analytics'.format(crayons.red('red')))
        fig_cgp = plt.figure(figsize=(10,8))
        fig_cgp, ax1 = plt.subplots(1,1)
        ax2 = ax1.twinx()
        ax1.plot(df["num_eval"],df["evaluation"],linestyle="solid",lw=0.5,ms=1,color="k",marker="^",label="loss")
        ax1.bar(df["num_eval"],df["size"],color="lightblue",label="size")
        ax1.set_ylim(0,20)
        ax2.set_ylim(1e10,1e11)
        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        ax1.legend(handler1+handler2,label1+label2,borderaxespad=0)
        #ax1.grid(True)
        fig_cgp.savefig(save_path+"/_log_cgp.png")

    def analytics_log_epoch_test(self,df,save_path):
        print('_log_epoch analytics'.format(crayons.red('red')))
        training_df = df.query('type == "training"')
        e_5 = training_df.query('epoch == 5')
        e_10 = training_df.query('epoch == 10')
        e_20 = training_df.query('epoch == 20')
        e_30 = training_df.query('epoch == 30')
        fig_loss = plt.figure(figsize=(10,8))
        x_ = [i+1 for i in range(len(e_5))]
        plt.plot(x_,e_5["test_loss"],lw=0.5,ms=1,label="5epoch")
        #plt.plot(x_,e_10["test_loss"],lw=0.5,ms=1,label="10epoch")
        #plt.plot(x_,e_20["test_loss"],lw=0.5,ms=1,label="20epoch")
        plt.plot(x_,e_30["test_loss"],lw=0.5,ms=1,label="30epoch")
        plt.ylim(0,100)
        plt.legend()
        fig_loss.savefig(save_path+"/_log_epoch_test_loss.png")

        fig_acc = plt.figure(figsize=(10,8))
        x_ = [i+1 for i in range(len(e_5))]
        plt.plot(x_,e_5["test_acc"],lw=1,ms=3,label="5epoch")
        #plt.plot(x_,e_10["test_acc"],lw=1,ms=3,label="10epoch")
        #plt.plot(x_,e_20["test_acc"],lw=1,ms=3,label="20epoch")
        plt.plot(x_,e_30["test_acc"],lw=1,ms=3,label="30epoch")
        plt.ylim(0,1)
        plt.legend()
        fig_acc.savefig(save_path+"/_log_epoch_test_acc.png")

    def analytics_log_local_refinement_best(self,df,save_path):
        print('_log_local_refinement_best analytics'.format(crayons.red('red')))
        for gen in range(df["generation"][-1]):
            df_epoch = df.query('type == "training"')
        fig = plt.figure(figsize=(10,8))