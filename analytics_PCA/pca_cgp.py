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
from WLM import WeisfeilerLehmanMachine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from archive import Archive
from scipy.interpolate import Rbf
from operator import attrgetter
from scipy.stats import rankdata
from scipy.stats import kendalltau

# Population of CGP
# gene[f][c] f:function type, c:connection (nodeID)
class Individual(object):
    # init -> False
    def __init__(self, net_info, init):
        self.net_info = net_info
        self.gene = np.zeros((self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)  # all True
        self.is_pool = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = None
        self.f_rbf = None
        self.size = None
        self.graph = nx.Graph()
        self.feature = []       # W2V
        # 初期化はFalse
        if init:
            print('init with specific architectures')
            self.init_gene_with_conv()  # In the case of starting only convolution
        else:
            self.init_gene()  # generate initial individual randomly

    def init_gene_with_conv(self):
        # initial architecture
        arch = ['S_ConvBlock_64_3']
        input_layer_num = int(self.net_info.input_num / self.net_info.rows) + 1
        output_layer_num = int(self.net_info.out_num / self.net_info.rows) + 1
        layer_ids = [((self.net_info.cols - 1 - input_layer_num - output_layer_num) + i) // (len(arch)) for i in
                     range(len(arch))]
        prev_id = 0  # i.e. input layer
        current_layer = input_layer_num
        block_ids = []  # *do not connect with these ids

        # building convolution net
        for i, idx in enumerate(layer_ids):
            current_layer += idx
            n = current_layer * self.net_info.rows + np.random.randint(self.net_info.rows)
            block_ids.append(n)
            self.gene[n][0] = self.net_info.func_type.index(arch[i])
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0

            self.gene[n][1] = prev_id
            for j in range(1, self.net_info.max_in_num):
                self.gene[n][j + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

            prev_id = n + self.net_info.input_num

        # output layer
        n = self.net_info.node_num
        type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
        self.gene[n][0] = np.random.randint(type_num)
        col = np.min((int(n / self.net_info.rows), self.net_info.cols))
        max_connect_id = col * self.net_info.rows + self.net_info.input_num
        min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
            if col - self.net_info.level_back >= 0 else 0

        self.gene[n][1] = prev_id
        for i in range(1, self.net_info.max_in_num):
            self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)
        block_ids.append(n)

        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):
            if n in block_ids:
                continue
            # type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

        self.check_active()

    def init_gene(self):    # finctionと接続ノードについての乱数をここで振り分ける
        # intermediate node 各ノードに対してコネクト判定を搭載していく
        for n in range(self.net_info.node_num + self.net_info.out_num):
            # type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            self.gene[n][0] = np.random.randint(type_num)   # functionIDをrandで設計
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

        self.check_active()


    def __check_course_to_out(self, n): # 関数の意義(n=150) 有効な接続をactiveに変更する関数
        if not self.is_active[n]:
            self.is_active[n] = True    # outputは強制的にアクティブにする
            t = self.gene[n][0]
            if n >= self.net_info.node_num:  # output node
                in_num = self.net_info.out_in_num[t]
            else:  # intermediate node
                in_num = self.net_info.func_in_num[t]

            for i in range(in_num): # おそらくoutputノードのarityの数の話をしている
                if self.gene[n][i + 1] >= self.net_info.input_num:  # 定義から引数の特定までもっていく
                    self.__check_course_to_out(self.gene[n][i + 1] - self.net_info.input_num)
            

    def check_active(self):
        # すべてをFalseに変更する
        self.is_active[:] = False
        # start from output nodes
        for n in range(self.net_info.out_num):
            self.__check_course_to_out(self.net_info.node_num + n)

    def check_pool(self):   # 関数の意義とは？
        is_pool = True
        pool_num = 0
        for n in range(self.net_info.node_num + self.net_info.out_num):
            if self.is_active[n]:   # activeなノードだけスクリーニングしておく
                if self.gene[n][0] > 19:
                    is_pool = False
                    pool_num += 1
        return is_pool, pool_num

    def __mutate(self, current, min_int, max_int):
        mutated_gene = current
        while current == mutated_gene:
            mutated_gene = min_int + np.random.randint(max_int - min_int)
        return mutated_gene

    def mutation(self, mutation_rate=0.01):
        active_check = False

        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if np.random.rand() < mutation_rate and max_connect_id - min_connect_id > 1:
                    self.gene[n][i + 1] = self.__mutate(self.gene[n][i + 1], min_connect_id, max_connect_id)
                    if self.is_active[n] and i < in_num:
                        active_check = True

        self.check_active()
        return active_check

    def neutral_mutation(self, mutation_rate=0.01):
        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if not self.is_active[n] and np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if (not self.is_active[n] or i >= in_num) and np.random.rand() < mutation_rate \
                        and max_connect_id - min_connect_id > 1:
                    self.gene[n][i + 1] = self.__mutate(self.gene[n][i + 1], min_connect_id, max_connect_id)

        self.check_active()
        return False

    def count_active_node(self):
        return self.is_active.sum()

    def copy(self, source):
        self.net_info = source.net_info
        self.gene = source.gene.copy()
        self.is_active = source.is_active.copy()
        self.eval = source.eval
        self.size = source.size

    def active_net_list(self):  # 可視化ツール的な役割を担う
        net_list = [["input", 0, 0]]
        active_cnt = np.arange(self.net_info.input_num + self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active)    # input以外は累積和を計算している

        for n, is_a in enumerate(self.is_active):
            if is_a:    # 活性ノードの場合
                t = self.gene[n][0]
                if n < self.net_info.node_num:  # intermediate node
                    # type_str -> type strings
                    type_str = self.net_info.func_type[t]
                else:  # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i + 1]] for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        return net_list     # 小さいノード番号に振り戻して表記してくれている

    def active_net_int_list(self):
        net_list = [["input", 0, 0]]
        active_cnt = np.arange(self.net_info.input_num + self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active) 

        for n, is_a in enumerate(self.is_active):
            if is_a:    # 活性ノードの場合
                t = self.gene[n][0]
                if n < self.net_info.node_num:  # intermediate node
                    # type_str -> type strings
                    type_str = self.net_info.func_type[t]
                else:  # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i + 1]] for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        return net_list     # 小さいノード番号に振り戻して表記してくれている


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
    
    def _evaluation_val(self):
        print("### Validation Population ###")
        eval_flag = np.ones(self.cnf.num_val_pop)
        self.val_loss, evaluations_size = self._evaluation(self.Val_pop, eval_flag=eval_flag)
        return self.val_loss

    def _vector_val(self):
        print("### Validation Population ###")
        #eval_flag = np.ones(self.cnf.num_val_pop)
        #self.val_loss, evaluations_size = self._evaluation(self.Val_pop, eval_flag=eval_flag)
        #self.val_rank = rankdata(self.val_loss)
        self.val_vector = []
        for pop in self.Val_pop:
            vector = self.trance_graph2vec(pop).tolist()
            if len(self.val_vector) == 0:
                self.val_vector = np.array([vector])
            else:
                self.val_vector = np.vstack((self.val_vector ,vector))
        return self.val_vector
    
    def _init_rank(self):
        self.val_loss = []
        for val_pop in self.Val_pop:
            self.val_loss.append(val_pop.eval)
        self.val_rank = rankdata(self.val_loss)
        self.val_vector = []
        for pop in self.Val_pop:
            vector = self.trance_graph2vec(pop).tolist()
            self.val_vector.append(vector)

    # 初期化の時には(pop = pop[0])だけが議論点だった
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
        # fp = [(評価値（acc）,パラメータ数)]
        fp = self.eval_func(net_lists)  # to __call__ (CNNEcaluation)   # 返り値は損失値
        for i, j in enumerate(active_index):
            if isinstance(fp[i], tuple):
                pop[j].eval = fp[i][0]
                pop[j].size = fp[i][1]
            else:
                pop[j].eval = fp[i]
                pop[j].size = np.inf
        evaluations_loss = np.zeros(len(pop))    # array([0.])
        evaluations_size = np.zeros(len(pop))
        for i in range(len(pop)):
            evaluations_loss[i] = pop[i].eval
            evaluations_size[i] = pop[i].size

        self.num_eval += len(net_lists)
        return evaluations_loss, evaluations_size

    # 親個体専門関数
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

    def _log_data_surrogate(self,kendalltau=0,rank_dif=0,rmse=0, start_time=0):
        log_list = {"generation":self.num_gen,"num_eval":self.num_eval,"time":time.time()-start_time,
                    "kendalltau":kendalltau,"rank_dif":rank_dif, "rmse":rmse}
        return log_list
                    
    # 子個体専門の関数
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
        print("### Start Initialize to G2V ###")
        document_collections = []
        for name , obj in tqdm(enumerate(self.Graph2Vec_pop)):
            document_collections.append(self.graph2doc(pop=obj,name=name))
        self.G2V = Doc2Vec(document_collections,
                            vector_size=self.cnf.vector_size,    # num_vector(128)
                            window=self.cnf.window,                           # 何単語でベクトル化するか
                            min_count=self.cnf.min_count,           # 指定の回数以下の出現回数の単語は無視する
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
                self._evaluation([self.Graph2Vec_pop[i]], np.array([True]))
                vector = self.trance_graph2vec(self.Graph2Vec_pop[i])
                self.Archive._add(var=vector,obj=self.Graph2Vec_pop[i].eval)
        self.Archive._stack()
        self.rbf = Rbf(*self.Archive.archive_stack, function='cubic')

        self.pop[0].copy(sorted(self.Graph2Vec_pop[:self.cnf.vector_size+1],key=attrgetter("eval"),reverse=False)[:1][-1])

        # sry , this is magic number
        for i in range(self.cnf.num_best_pop):
            self.Best_pop[i].copy(sorted(self.Graph2Vec_pop[:self.cnf.vector_size+1],key=attrgetter("eval"),reverse=False)[:self.cnf.num_best_pop][i])
        self.df_cgp_log.append(self._log_data(net_info_type='no_gene', start_time=start_time, pop=self.pop[0]))

        eval_flag = np.empty(self.lam)
        while self.num_gen < max_eval:
            start_time = time.time()            # 連続構造
            self.num_gen += 1
            # reproduction
            # local refinement
            print("### LocalRefinement ###")
            print("---GENERATION : ",self.num_gen)
            for local_i in tqdm(range(self.cnf.localrefinement)):
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
                f_rbf = self.rbf(*(np.array(x_).T))
                for i in range(self.lam):
                    self.pop[i+1].f_rbf = f_rbf[i]

                self.pop[0].copy(sorted(self.pop[1:],key=attrgetter("f_rbf"),reverse=False)[:1][-1])

            # kendall tau
            rbf_acc = self.rbf(*(np.array(self.val_vector).T))
            rbf_rank = rankdata(rbf_acc)

            rank_dif = np.abs(rbf_rank - self.val_rank)
            rank_dif = np.sum(rank_dif)
            correlation, pvalue = kendalltau(rbf_rank,self.val_rank)
            rmse = np.sqrt(np.mean((self.val_loss - rbf_acc)**2))
            self.df_surrogate_log.append(self._log_data_surrogate(kendalltau=correlation, rank_dif=rank_dif, rmse=rmse,start_time=start_time))
            self._log_surrogate_save()
            self.df_archive_log = self.Archive._log_archive(self.num_eval)
            self._log_archive_save()

            # evaluation and selection
            self.pop_cand = sorted(self.pop[1:],key=attrgetter("f_rbf"),reverse=False)
            self.cand_loss, evaluations_size = self._evaluation(self.pop_cand[:self.cnf.num_best_pop], eval_flag=eval_flag[:self.cnf.num_best_pop])

            for i in range(self.cnf.num_best_pop):
                vector = self.trance_graph2vec(self.pop[i])
                self.Archive._add(var=vector,obj=self.pop[i].eval)
            self.rbf = Rbf(*self.Archive.archive_stack, function='cubic')
            self.Archive._sort()

            # RBF sort
            if self.pop[0].eval <= self.Best_pop[0].eval:
                self.Best_pop[0].copy(self.pop[0])
            else:
                self.pop[0].neutral_mutation(mutation_rate)

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
