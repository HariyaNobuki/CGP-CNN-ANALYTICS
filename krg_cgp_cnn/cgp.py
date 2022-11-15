#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import time
import numpy as np
import math
import pandas as pd
import os
from sklearn.feature_selection import SelectFdr
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

### My module
from krg_cgp_cnn.WLM import WeisfeilerLehmanMachine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from krg_cgp_cnn.archive import Archive
from scipy.interpolate import Rbf
from operator import attrgetter
from scipy.stats import rankdata
from scipy.stats import kendalltau , norm
import crayons
from    pydacefit.dace                      import DACE
import  GPy

# split module
from krg_cgp_cnn.indinidual import Individual


# CGP with (1 + \lambda)-ES
class CGP(object):
    def __init__(self, net_info, eval_func, cnf, lam=4, imgSize=32, init=False, bias=0, G2V_pop=100):
        self.lam = lam
        self.pop = [Individual(net_info, init) for _ in range(1 + self.lam)]    # サロゲート回転用
        self.eval_func = eval_func
        self.num_gen = 0
        self.num_eval = 0
        self.max_pool_num = int(math.log2(imgSize) - 2) # 3
        self.init = init
        self.bias = bias
        self.cnf = cnf
        # My custom
        self.Graph2Vec_pop = [Individual(net_info, init) for _ in range(G2V_pop)]
        self.Archive = Archive()
        self.Best_pop = [Individual(net_info, init) for _ in range(self.cnf.num_best_pop)]
        self.Val_pop = [Individual(net_info, init) for _ in range(self.cnf.num_val_pop)]
        self.pop_cand = [Individual(net_info, init) for _ in range(self.cnf.num_best_pop)]
    
    def surrogate_epoch(self):
        self.cnf.epoch_num = self.cnf.sur_epoch
    
    def long_cnn_epock(self):
        self.cnf.epoch_num = self.cnf.full_epoch

    def _evaluation_val(self):  # always long
        print(crayons.blue("### Validation Population ###"))
        eval_flag = np.ones(self.cnf.num_val_pop)
        self.val_loss , param = self._evaluation(self.Val_pop, eval_flag=eval_flag)
        val_evaluations = []
        for pop in self.Val_pop:
            val_evaluations.append(pop.eval)
        self.val_rank = rankdata(val_evaluations)
        # independent graph node
        self.val_vector = []
        for pop in self.Val_pop:
            vector = self.trance_graph2vec(pop).tolist()
            self.val_vector.append(vector)

    # self._evaluation([self.pop[0]], np.array([True]))
    def _evaluation(self, pop, eval_flag):
        # create network list
        net_lists = []
        active_index = np.where(eval_flag)[0]
        for i in active_index:
            net_lists.append(pop[i].active_net_list())

        # create network int list
        net_int_lists = []
        for i in active_index:
            net_int_lists.append(pop[i].active_net_int_list())

        # evaluation
        # net_lists -> activeなノードの配列構造
        fp = self.eval_func(net_lists)  # to __call__ (CNNEcaluation)   # 返り値は損失値
        for i, j in enumerate(active_index):
            if isinstance(fp[i], tuple):    # default
                pop[j].eval_list = fp[i][2]
                pop[j].eval = fp[i][2][-1][self.cnf.custom_loss]
                pop[j].size = fp[i][1]
            else:
                pop[j].eval_list = fp[i]
                pop[j].eval_list = fp[i][-1][self.cnf.custom_loss]
                pop[j].size = np.inf
        evaluations_loss = np.zeros(len(pop))    # array([0.])
        evaluations_size = np.zeros(len(pop))
        for i in range(len(pop)):
            evaluations_loss[i] = pop[i].eval
            evaluations_size[i] = pop[i].size

        self.num_eval += len(net_lists)
        return evaluations_loss, evaluations_size

    def _log_data(self, net_info_type='active_only', start_time=0, pop=None):
        log_list = {"generation":self.num_gen,"num_eval":self.num_eval,"time":time.time()-start_time,
                    "evaluation":pop.eval,"size":pop.size, "active_node":pop.count_active_node()}
        if net_info_type == 'active_only':
            log_list.append(pop.active_net_list())
        elif net_info_type == 'full':
            log_list += pop.gene.flatten().tolist()
        elif net_info_type == 'no_gene':
            pass
        return log_list

    def _log_best_local_data(self,local_i=None,sur_original=None, start_time=0, f_sur=None,x_=None):
        min_idx = np.argmin(f_sur)
        dec_var = x_[min_idx]
        log_list = {"generation":self.num_gen,"num_eval":self.num_eval,"local_i":local_i,"time":time.time()-start_time,
                    "f_sur":f_sur[min_idx],"f_IC":sur_original[min_idx]}
        for dim in range(len(dec_var)):
            log_list["x{}".format(dim)] = dec_var[dim]
        return log_list

    def _log_local_data(self,sur_original=None, start_time=0, f_sur=None,x_=None):
        log_list = {"generation":self.num_gen,"num_eval":self.num_eval,"time":time.time()-start_time,
                "f_sur":f_sur,"f_IC":sur_original}
        for dim in range(len(x_)):
            log_list["x{}".format(dim)] = x_[dim]
        return log_list

    def _log_data_surrogate(self,kendalltau=0,rank_dif=0,rmse=0, start_time=0):
        log_list = {"generation":self.num_gen,"num_eval":self.num_eval,"time":time.time()-start_time,
                    "kendalltau":kendalltau,"rank_dif":rank_dif, "rmse":rmse}
        return log_list

    def _log_data_children(self, net_info_type='active_only', start_time=0, pop=None):
        log_list = [self.num_gen, self.num_eval, time.time() - start_time, pop.eval, pop.size, pop.count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(pop.active_net_list())
        elif net_info_type == 'full':
            log_list += pop.gene.flatten().tolist()
        elif net_info_type == 'no_gene':
            pass
        return log_list

    def load_log(self, log_data):
        self.num_gen = int(log_data[0])
        self.num_eval = int(log_data[1])
        net_info = self.pop[0].net_info
        self.pop[0].eval = log_data[3]
        self.pop[0].size = log_data[4]
        print("Loaded Accuracy:", self.pop[0].eval)
        self.pop[0].gene = np.int64(np.array(log_data[6:])).reshape(
            (net_info.node_num + net_info.out_num, net_info.max_in_num + 1))
        self.pop[0].check_active()
    
    def G2V_initialization(self):
        # produce Val_pop timinig is cgp intializer
        print(crayons.blue("### Start Initialize to"),end="")
        print(crayons.red(" G2V ###"))
        document_collections = []
        for name , obj in tqdm(enumerate(self.Graph2Vec_pop)):
            document_collections.append(self.graph2doc(pop=obj,name=name))
        self.G2V = Doc2Vec(document_collections,
                            vector_size=self.cnf.vector_size, 
                            window=self.cnf.window,       
                            min_count=self.cnf.min_count, 
                            dm=self.cnf.dm,
                            sample=self.cnf.sample,
                            workers=self.cnf.workers,
                            epochs=self.cnf.epochs,
                            alpha=self.cnf.alpha)

    def graph2doc(self,pop,name):
        # initialization
        edge_list = []
        feature_dict = {}
        for node in range(pop.gene.shape[0]):
            feature_dict[node] = str(pop.gene[node][0])
            pop.graph.add_node(node)
            if pop.is_active[node]: # EDGE
                if node == pop.gene.shape[0]-1:
                    edge_list.append((node,pop.gene[node][1]))
                    edge_list.append((node,pop.gene[node][2]))
                else:
                    edge_list.append((node+1,pop.gene[node][1]))
                    edge_list.append((node+1,pop.gene[node][2]))
        pop.graph.add_edges_from(edge_list)
        machine = WeisfeilerLehmanMachine(pop.graph, feature_dict, 1) # arity2 is dependence
        doc = TaggedDocument(words=machine.extracted_features, tags=["g_{}".format(name)])
        return doc
    
    def trance_graph2vec(self,pop):
        # Sample Predict(EX)
        return self.G2V.infer_vector(self.graph2doc(pop,name=101)[0])    # ここのnameはなんでもよさそう

    # Evolution CGP:
    #   At each iteration:
    #     - Generate lambda individuals in which at least one active node changes (i.e., forced mutation)
    #     - Mutate the best individual with neutral mutation (unchanging the active nodes)
    #         if the best individual is not updated.
    def modified_evolution(self, max_eval=100, mutation_rate=0.01, log_path='./'):
        self.num_eval = 0           # indeodendance of validation data
        self.df_cgp_log = []
        self.df_surrogate_log = []

        # for RBF , add eval count
        for i in range(self.cnf.vector_size+1):
            start_time = time.time()
            active_num = self.Graph2Vec_pop[i].count_active_node()
            _, pool_num = self.Graph2Vec_pop[i].check_pool()
            if self.init:   # init -> False
                pass
            else:  # in the case of not using an init indiviudal
                while active_num < self.Graph2Vec_pop[i].net_info.min_active_num or pool_num > self.max_pool_num:
                    self.Graph2Vec_pop[i].mutation(1.0)
                    active_num = self.Graph2Vec_pop[i].count_active_node()
                    _, pool_num = self.Graph2Vec_pop[i].check_pool()
            if self.Graph2Vec_pop[i].eval is None:
                if self.cnf.EMultiMode:
                    self.surrogate_epoch()
                else:
                    self.long_cnn_epock()
                self._evaluation([self.Graph2Vec_pop[i]], np.array([True]))
                vector = self.trance_graph2vec(self.Graph2Vec_pop[i])
                self.Archive._add(var=vector,obj=self.Graph2Vec_pop[i].eval)
        self.Archive._stack()
        self.Archive._sort()
        # surrogate init
        if self.cnf.sur_mode == "RBF":
            self.sur = Rbf(*self.Archive.archive_stack, function='cubic')
        elif self.cnf.sur_mode == "KRG":
            theta0 = self.cnf.dace_theta
            self.sur = DACE(regr=self.cnf.dace_regr, corr=self.cnf.dace_corr, theta=np.full(self.cnf.vector_size, theta0), thetaL=[self.cnf.dace_thetaL] * self.cnf.vector_size, thetaU=[self.cnf.dace_thetaU] * self.cnf.vector_size)
            self.sur.fit(self.Archive.archive_var, self.Archive.archive_obj)

        self.pop[0].copy(sorted(self.Graph2Vec_pop[:self.cnf.vector_size+1],key=attrgetter("eval"),reverse=False)[:1][-1])

        # sry , this is magic number
        for i in range(self.cnf.num_best_pop):
            self.Best_pop[i].copy(sorted(self.Graph2Vec_pop[:self.cnf.vector_size+1],key=attrgetter("eval"),reverse=False)[:self.cnf.num_best_pop][i])
        self.df_cgp_log.append(self._log_data(net_info_type='no_gene', start_time=start_time, pop=self.pop[0]))

        eval_flag = np.empty(self.lam)
        while self.num_gen < max_eval:
            start_time = time.time()    # get cwd time
            self.num_gen += 1
            # reproduction
            # local refinement
            print(crayons.red("---GENERATION : "),self.num_gen)
            self.df_local_refinement_best_log = []
            self.df_local_refinement_log = []
            for local_i in tqdm(range(self.cnf.localrefinement)):
                start_local_time = time.time()
                x_ = []             # cand_vec
                for i in range(self.lam):
                    eval_flag[i] = False
                    self.pop[i + 1].copy(self.pop[0])  # copy a parent copy
                    active_num = self.pop[i + 1].count_active_node()
                    _, pool_num = self.pop[i + 1].check_pool()
                    # mutation (forced mutation)
                    while not eval_flag[i] or active_num < self.pop[i + 1].net_info.min_active_num or pool_num > self.max_pool_num:
                        self.pop[i + 1].copy(self.pop[0])  # copy a parent
                        eval_flag[i] = self.pop[i + 1].mutation(mutation_rate)  # mutation
                        active_num = self.pop[i + 1].count_active_node()
                        _, pool_num = self.pop[i + 1].check_pool()
                    # graph2vec
                    if len(x_)==0:
                        x_.append(self.trance_graph2vec(self.pop[i+1]).tolist())
                    else:
                        #if not np.any(np.all(x_ == self.trance_graph2vec(self.pop[i+1]), axis=1)):  # もしも重複したら
                        x_.append(self.trance_graph2vec(self.pop[i+1]).tolist())
                if self.cnf.sur_mode == "RBF":
                    f_sur = self.sur(*(np.array(x_).T))
                    IC = np.zeros(len(f_sur))
                    for i in range(self.lam):
                        self.pop[i+1].f_sur = f_sur[i]
                    self.pop[0].copy(sorted(self.pop[1:],key=attrgetter("f_sur"),reverse=False)[:1][-1])
                elif self.cnf.sur_mode == "KRG":
                    f_sur, _pseudo_var = self.sur.predict(np.array(x_), return_mse=True)
                    pred_std = np.sqrt(_pseudo_var)
                    f_sur, pred_std  = f_sur.reshape(1,-1)[0], pred_std.reshape(1,-1)[0]
                    IC = np.zeros(len(f_sur))
                    Gbest   = np.min(self.Archive.archive_obj[0])
                    mask    = ~((pred_std == 0) | np.isnan(pred_std))
                    IC[mask]= (Gbest-f_sur[mask]) * norm.cdf((Gbest-f_sur[mask])/pred_std[mask]) + pred_std[mask] * norm.pdf((Gbest-f_sur[mask])/pred_std[mask])
                    for i in range(self.lam):
                        self.pop[i+1].f_sur = f_sur[i]
                        self.pop[i+1].f_IC = IC[i]
                    self.pop[0].copy(sorted(self.pop[1:],key=attrgetter("f_IC"),reverse=True)[:1][-1])

                # local refinement saving
                if local_i % 100 == 0:
                    self.df_local_refinement_best_log.append(self._log_best_local_data(local_i=local_i,sur_original=IC,start_time=start_local_time,f_sur=f_sur,x_=x_))
                    self._log_local_refinement_best_save()
            for i in range(self.cnf.lam_save):
                self.df_local_refinement_log.append(self._log_local_data(sur_original=IC[i],start_time=start_local_time,f_sur=f_sur[i],x_=x_[i]))
            self._log_local_refinement_save()


            # kendall tau
            if self.cnf.sur_mode == "RBF":
                val_f = self.sur(*(np.array(self.val_vector).T))
            elif self.cnf.sur_mode == "KRG":
                val_f, _pseudo_var = self.sur.predict(np.array(self.val_vector), return_mse=True)
            val_rank = rankdata(val_f)

            rank_dif = np.abs(val_rank - self.val_rank)
            rank_dif = np.sum(rank_dif)
            correlation, pvalue = kendalltau(val_rank,self.val_rank)
            rmse = np.sqrt(np.mean((self.val_loss - val_f)**2))
            self.df_surrogate_log.append(self._log_data_surrogate(kendalltau=correlation, rank_dif=rank_dif, rmse=rmse,start_time=start_time))
            self._log_surrogate_save()
            self.df_archive_log = self.Archive._log_archive(self.num_eval)
            self._log_archive_save()

            # evaluation and selection
            if self.cnf.sur_mode == "RBF":
                for i in range(self.cnf.num_best_pop):
                    self.pop_cand[i].copy(sorted(self.pop[1:],key=attrgetter("f_sur"),reverse=False)[i])
            elif self.cnf.sur_mode == "KRG":
                for i in range(self.cnf.num_best_pop):
                    self.pop_cand[i].copy(sorted(self.pop[1:],key=attrgetter("f_IC"),reverse=True)[i])
            self.long_cnn_epock()
            self.cand_loss = self._evaluation(self.pop_cand[:self.cnf.num_best_pop], eval_flag=eval_flag[:self.cnf.num_best_pop])

            # GBest sort
            if sorted(self.pop_cand,key=attrgetter("eval"),reverse=False)[0].eval <= self.Best_pop[0].eval:
                self.Best_pop[0].copy(sorted(self.pop_cand,key=attrgetter("eval"),reverse=False)[0])
            else:
                self.pop[0].neutral_mutation(mutation_rate)
            

            if self.cnf.EMultiMode:
                for i in range(self.cnf.num_best_pop):
                    vector = self.trance_graph2vec(self.pop_cand[i])
                    self.Archive._add(var=vector,obj=pd.DataFrame(self.pop_cand[i].eval_list).query("epoch=={}".format(self.cnf.sur_epoch))[self.cnf.custom_loss])
                if self.cnf.sur_mode == "RBF":
                    self.sur = Rbf(*self.Archive.archive_stack, function='cubic')
                elif self.cnf.sur_mode == "KRG":
                    theta0 = self.cnf.dace_theta
                    self.sur = DACE(regr=self.cnf.dace_regr, corr=self.cnf.dace_corr, theta=np.full(self.cnf.vector_size, theta0), thetaL=[self.cnf.dace_thetaL] * self.cnf.vector_size, thetaU=[self.cnf.dace_thetaU] * self.cnf.vector_size)
                    self.sur.fit(self.Archive.archive_var, self.Archive.archive_obj)
                self.Archive._sort()
            else:
                for i in range(self.cnf.num_best_pop):
                    vector = self.trance_graph2vec(self.pop_cand[i])
                    self.Archive._add(var=vector,obj=pd.DataFrame(self.pop_cand[i].eval_list).query("epoch=={}".format(self.cnf.full_epoch))[self.cnf.custom_loss])
                if self.cnf.sur_mode == "RBF":
                    self.sur = Rbf(*self.Archive.archive_stack, function='cubic')
                elif self.cnf.sur_mode == "KRG":
                    theta0 = self.cnf.dace_theta
                    self.sur = DACE(regr=self.cnf.dace_regr, corr=self.cnf.dace_corr, theta=np.full(self.cnf.vector_size, theta0), thetaL=[self.cnf.dace_thetaL] * self.cnf.vector_size, thetaU=[self.cnf.dace_thetaU] * self.cnf.vector_size)
                    self.sur.fit(self.Archive.archive_var, self.Archive.archive_obj)
                self.Archive._sort()

            self.df_cgp_log.append(self._log_data(net_info_type='no_gene', start_time=start_time, pop=self.Best_pop[0]))

            self._log_save()


    def _log_save(self):
        df_n = pd.DataFrame(self.df_cgp_log)        # new
        self.df_cgp_log = []
        if os.path.isfile(self.cnf.trial_path + "/_log_cgp.csv"):
            df_o = pd.read_csv(self.cnf.trial_path + "/_log_cgp.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.trial_path + "/_log_cgp.csv",index = False)
        else:
            df_n.to_csv(self.cnf.trial_path + "/_log_cgp.csv",index = False)

    def _log_archive_save(self):
        df_n = pd.DataFrame(self.df_archive_log)        # new
        self.df_archive_log = []
        if os.path.isfile(self.cnf.trial_path + "/_log_archive.csv"):
            df_o = pd.read_csv(self.cnf.trial_path + "/_log_archive.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.trial_path + "/_log_archive.csv",index = False)
        else:
            df_n.to_csv(self.cnf.trial_path + "/_log_archive.csv",index = False)

    def _log_surrogate_save(self):
        df_n = pd.DataFrame(self.df_surrogate_log)        # new
        self.df_surrogate_log = []
        if os.path.isfile(self.cnf.trial_path + "/_log_surrogate.csv"):
            df_o = pd.read_csv(self.cnf.trial_path + "/_log_surrogate.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.trial_path + "/_log_surrogate.csv",index = False)
        else:
            df_n.to_csv(self.cnf.trial_path + "/_log_surrogate.csv",index = False)

    def _log_local_refinement_best_save(self):
        df_n = pd.DataFrame(self.df_local_refinement_best_log)        # new
        self.df_local_refinement_best_log = []
        if os.path.isfile(self.cnf.trial_path + "/_log_local_refinement_best.csv"):
            df_o = pd.read_csv(self.cnf.trial_path + "/_log_local_refinement_best.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.trial_path + "/_log_local_refinement_best.csv",index = False)
        else:
            df_n.to_csv(self.cnf.trial_path + "/_log_local_refinement_best.csv",index = False)

    def _log_local_refinement_save(self):
        df_n = pd.DataFrame(self.df_local_refinement_log)        # new
        self.df_local_refinement_log = []
        if os.path.isfile(self.cnf.trial_path + "/_log_local_refinement.csv"):
            df_o = pd.read_csv(self.cnf.trial_path + "/_log_local_refinement.csv")
            df_m = pd.concat([df_o,df_n],axis=0)        # merge
            df_m.to_csv(self.cnf.trial_path + "/_log_local_refinement.csv",index = False)
        else:
            df_n.to_csv(self.cnf.trial_path + "/_log_local_refinement.csv",index = False)
