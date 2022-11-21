import crayons  # print('{}'.format(crayons.red('red')))
import random
import os , sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Analytics:
    def __init__(self):
        print(crayons.red("### ANALYTICS ###"))
    
    def analytics_log_cgp(self):
        print(crayons.blue("### CGP ###"))
        fig_cgp = plt.figure(figsize=(10,8))
        fig, ax1 = plt.subplots(1,1)
        ax2 = ax1.twinx()
