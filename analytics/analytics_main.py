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
        ax2.set_ylim(10000000,100000000)
        handler1, label1 = ax1.get_legend_handles_labels()
        handler2, label2 = ax2.get_legend_handles_labels()
        ax1.legend(handler1+handler2,label1+label2,borderaxespad=0)
        #ax1.grid(True)
        fig_cgp.savefig(save_path+"/_log_cgp.png")