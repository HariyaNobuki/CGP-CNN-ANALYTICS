# python 3.7 plz
import crayons
import random
import os , sys
import numpy as np
import torch

from   pydacefit.corr   import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
from   pydacefit.regr   import regr_constant, regr_linear, regr_quadratic

class Configuration:
    def __init__(self):
        print(crayons.green("### CONFIG Initialization ###"))
        self.num_trial = 3
        self.main_path = os.getcwd()
        self.gpu_num = 1
        self.res_path = self.main_path+"/result"

        self.CGP_mode = "evolution"
        self.sur_mode = "RBF"       # ["KRG","RBF"]
        self.EMultiMode = False

        self.sub_mode = "analytics"

    
    def setting(self):
        self.max_eval = 60
        self.mutation_rate = 0.1
    
    def set_ANA(self):
        self.res_date = "20221122"
        self.analytics_mode = ["_log_cgp"]
        self.exp_mode = ["RBF-F","RBF-T","KRG-d5-F","KRG-d5-T","KRG-d10-F","KRG-d10-T"]
        self.nas_path = os.path.expanduser(r'~/Desktop/192.168.11.8\Experiment\2023_hariya')

    def set_G2V(self):
        self.vector_size= 10
        self.window=20
        self.min_count=1
        self.dm=0
        self.sample=5     
        self.workers=5    
        self.epochs=2000
        self.alpha=0.025

    def set_CNN(self):
        self.epoch_num = 30
        self.reduced = True
        self.sur_epoch = 5
        self.full_epoch = 30
        # custom selection "test_loss","test_acc"
        self.custom_loss = "test_loss"
    

    def set_CGP(self):
        # grid
        self.rows = 2
        self.cols = 20
        self.level_back = 10
        self.min_active_num = 1
        self.max_active_num = 20

        self.lam = 100
        self.lam_save = 10
        self.imgSize = 32
        self.init = False
        self.bias = 0
        self.G2V_pop = 2000

        self.localrefinement = 1000

        self.dace_regr      = regr_constant
        self.dace_corr      = corr_gauss
        self.dace_theta = 1.                 #0.01
        self.dace_thetaL    = 0.00001
        self.dace_thetaU    = 100.

        self.num_best_pop = 3
        self.margin_val_pop = 2
        self.num_val_pop = 25

    def set_analitics(self):
        self.ANA_CGP = True
        self.ANA_CNN = True
        self.ANA_SUR = True

    def torch_fix_seed(self,seed):
        # Python random
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # Pytorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True