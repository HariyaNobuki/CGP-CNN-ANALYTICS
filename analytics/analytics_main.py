import crayons  # print('{}'.format(crayons.red('red')))
import random
import os , sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Analytics:
    def __init__(self):
        print(crayons.red("### ANALYTICS ###"))
    
    def analytics_log_cgp(self,df):
        print('cgp analytics'.format(crayons.red('red')))
        fig_cgp = plt.figure(figsize=(10,8))
        fig, ax1 = plt.subplots(1,1)
        ax2 = ax1.twinx()
        ax1.plot(df["num_eval"],df["evaluation"],linestyle="solid",color="k",marker="^",label="loss")
        ax1.bar(df["num_eval"],df["size"],color="lightblue",label="size")
        #ax1.set_ylim(0,10)
        #ax2.set_ylim(100,110)
