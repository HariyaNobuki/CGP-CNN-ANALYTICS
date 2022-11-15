import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE


import os , sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))  # 1階層上のfileのimport
import configuration

# Original Module
sys.path.append(os.path.join(os.path.dirname(__file__), '../analytics_PCA'))  # 1階層上のfileのimport
from pca_cgp import *
from pca_cgp_config import *
from pca_cnn_train import CNN_train

def maketrialfiles():
    for trial in range(cnf.num_trial):
        os.makedirs(cnf.res_path+"/trial_{}".format(trial),exist_ok=True)

if __name__ == "__main__":
    cnf = configuration.Configuration()
    cnf.setting()
    main_path = os.getcwd()
    cnf.trial_path = main_path + "/analytics_PCA"
    os.makedirs(main_path+"/PCA",exist_ok=True)
    cnf.torch_fix_seed(0)
    network_info = CgpInfoConvSet(rows=2, cols=5, level_back=5, min_active_num=1, max_active_num=5)
    imgSize = 32
    cnf.set_CNN()
    eval_f = CNNEvaluation(cnf,gpu_num=cnf.gpu_num, dataset='cifar10', reduced=cnf.reduced, verbose=True, epoch_num=cnf.epoch_num, 
                        batchsize=128*2, imgSize=imgSize)  # In cgp_config
    cnf.set_G2V()
    cnf.set_CGP()
    cnf.num_val_pop = 4
    cnf.vector_size = 10
    cnf.epoch_num = 1
    cgp = CGP(network_info, eval_f, cnf, lam=cnf.lam, imgSize=imgSize, init=cnf.init, bias=cnf.bias,G2V_pop=cnf.G2V_pop)
    cgp.G2V_initialization()
    #train_labels = cgp._evaluation_val()
    train_labels =[ 4.59629703, 49.35432053, 49.35432053, 49.35432053]
    val_vector = cgp._vector_val()
    df_train = pd.DataFrame(val_vector)
    print("===df_train.shape : ",df_train.shape)

    pca = PCA()
    pca.fit(df_train)
    feature = pca.transform(df_train)   # 主成分分析で変換 (60000,784)

    #二次元で可視化(おそらく強引に２次元におとしているはず)
    fig = plt.figure()
    sc = plt.scatter(feature[:,0],feature[:,1], vmin=-1, vmax=1,alpha=0.8,c=train_labels, cmap=cm.seismic)
    plt.colorbar(sc)
    plt.legend()
    fig.savefig(main_path+"/PCA/PCA.png")
    plt.clf()
    plt.close()


    print("--- explained_variance_ratio_ ---")
    #print(pca.explained_variance_ratio_)    # (784,)    各成分が持つ分散の比率
    print("--- components ---") # 主成分
    #print(pca.components_)  # (784 , 784) おそらく784成分の軸の向きを表している
    print("--- mean ---")       # 平均
    #print(pca.mean_)    # (784,)
    print("--- covariance ---") # 共分散
    #print(pca.get_covariance()) # (784 , 784)

    # いくつの成分を用いて適用するべきなのかを議論する方法
    ev_ratio = pca.explained_variance_ratio_    # これは何を表すのかはわからないけど分析の方法は割れてきた
    ev_ratio = np.hstack([0,ev_ratio.cumsum()])

    df_ratio = pd.DataFrame({"components":range(len(ev_ratio)), "ratio":ev_ratio})

    fig = plt.figure()
    plt.plot(ev_ratio)
    plt.xlabel("components")
    plt.ylabel("explained variance ratio")
    fig.savefig(main_path+"/PCA/NumCluster.png")
    plt.clf()
    plt.close()

    fig = plt.figure()
    plt.scatter(range(len(ev_ratio)),ev_ratio)
    fig.savefig(main_path+"/PCA/scat_evrat.png")
    plt.clf()
    plt.close()

    # ここまでは理解したで
    KM = KMeans(n_clusters = 3)
    result = KM.fit(feature[:,:2])
    #result.labels_ 予測ラベル

    df_eval = pd.DataFrame(confusion_matrix(train_labels,result.labels_))   # 混同行列
    df_eval.columns = df_eval.idxmax()  # 縦に予測している
    df_eval = df_eval.sort_index(axis=1)

    # ここで失敗を確認できる
    print(df_eval)

    #クラスタの中のデータの最も多いラベルを正解ラベルとしてそれが多くなるようなクラスタ数を探索
    eval_acc_list=[]

    for i in range(5,15):
        KM = KMeans(n_clusters = i)
        result = KM.fit(feature[:,:9])
        df_eval = pd.DataFrame(confusion_matrix(train_labels,result.labels_))
        eval_acc = df_eval.max().sum()/df_eval.sum().sum()
        eval_acc_list.append(eval_acc)

    fig = plt.figure()
    plt.plot(range(5,15),eval_acc_list)
    plt.xlabel("The number of cluster")
    plt.ylabel("accuracy")
    fig.savefig("Thenumberofcluster.png")
    plt.clf()
    plt.close()

    tsne = TSNE(n_components=2).fit_transform(feature[:10000,:9])
    fig = plt.figure(figsize=(8,5))
    for i in range(10):
        idx = np.where(train_labels[:10000]==i)
        plt.scatter(tsne[idx,0],tsne[idx,1],label=i)
    plt.legend(loc='upper left',bbox_to_anchor=(1.05,1))
    fig.savefig("tsne.png")
    plt.clf()
    plt.close()

    #tsneしたものをkmeansで分類
    KM = KMeans(n_clusters = 10)
    result = KM.fit(tsne)

    df_eval = pd.DataFrame(confusion_matrix(train_labels[:10000],result.labels_))
    df_eval.columns = df_eval.idxmax()
    df_eval = df_eval.sort_index(axis=1)

    print(df_eval)