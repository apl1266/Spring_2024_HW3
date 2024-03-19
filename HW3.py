import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import time
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score,adjusted_mutual_info_score, homogeneity_score,silhouette_score
from sklearn.decomposition import PCA,FastICA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.metrics import mean_squared_error
from sklearn.random_projection import GaussianRandomProjection
import os
import warnings
from plotting import *
from parter import *

warnings.filterwarnings("ignore")

np.random.seed(812)


def frames(name,rat):
    pd_fruits=pd.read_csv(name)
    if name=="fruits_data.csv":
        pd_fruits.loc[pd_fruits["Class"] == "BERHI", "Class"] = 0
        pd_fruits.loc[pd_fruits["Class"] == "DEGLET", "Class"] = 1
        pd_fruits.loc[pd_fruits["Class"] == "DOKOL", "Class"] = 2
        pd_fruits.loc[pd_fruits["Class"] == "IRAQI", "Class"] = 3
        pd_fruits.loc[pd_fruits["Class"] == "ROTANA", "Class"] = 4
        pd_fruits.loc[pd_fruits["Class"] == "SAFAVI", "Class"] = 5
        pd_fruits.loc[pd_fruits["Class"] == "SOGAY", "Class"] = 6

    scaler=preprocessing.MinMaxScaler()
    np_fruits=pd_fruits.values
    np_fruits_scaled=scaler.fit_transform(np_fruits)
    np.random.shuffle(np_fruits_scaled)
    if name == "fruits_data.csv":
        np_fruits_scaled[:,-1]=np_fruits_scaled[:,-1]*6
    else:
        np_fruits_scaled[:, -1] = np_fruits_scaled[:, -1] * 3
    split=int(np.shape(np_fruits_scaled)[0]*rat//1)
    np_fruits_scaled[:,-1] = np_fruits_scaled[:,-1].astype(int)
    return np_fruits_scaled[split:,:-1],np_fruits_scaled[split:,-1].flatten(),np_fruits_scaled[:split,:-1],np_fruits_scaled[:split,-1].flatten()





fruits_train_x,fruits_train_y,fruits_test_x,fruits_test_y= frames("fruits_data.csv",0)
phones_train_x,phones_train_y,phones_test_x,phones_test_y= frames("phones_data.csv",0)


t=time.time()
if 1:
    #part_1 runner
    a_0,b_0=part_1(fruits_train_x,fruits_train_y,"Fruits dataset",method="KMeans")
    a_1,b_1=part_1(fruits_train_x,fruits_train_y,"Fruits dataset",method="EM")
    a_2,b_2=part_1(phones_train_x,phones_train_y,"Phones dataset",method="KMeans")
    a_3,b_3=part_1(phones_train_x,phones_train_y,"Phones dataset",method="EM")
    plotter_23((a_0,b_0,b_1,a_2,b_2,b_3),"part_1.png")
    t0=time.time()
    KMeans(n_clusters=11, max_iter=2000, random_state=812, n_init=10).fit(fruits_train_x)
    print(time.time()-t0," seconds to run KMEANS on FRUITS data set")
    t0 = time.time()
    KMeans(n_clusters=4, max_iter=2000, random_state=812, n_init=10).fit(phones_train_x)
    print(time.time() - t0, " seconds to run KMEANS on Phones data set")

    t0 = time.time()
    GaussianMixture(n_components=10, max_iter=100, random_state=812,n_init=5).fit(fruits_train_x)
    print(time.time() - t0, " seconds to run EM on FRUITS data set")
    t0 = time.time()
    GaussianMixture(n_components=4, max_iter=100, random_state=812,n_init=5).fit(phones_train_x)
    print(time.time() - t0, " seconds to run EM on Phones data set")

if 1:
    # part_2 runner
    PCA_F=part_2(fruits_train_x,"Fruits dataset","PCA")
    ICA_F = part_2(fruits_train_x, "Fruits dataset", "ICA")
    RP_F = part_2(fruits_train_x, "Fruits dataset", "RP")
    LLE_F = part_2(fruits_train_x, "Fruits dataset", "MLLE")

    PCA_P = part_2(phones_train_x, "Phones dataset", "PCA")
    ICA_P = part_2(phones_train_x, "Phones dataset", "ICA")
    RP_P = part_2(phones_train_x, "Phones dataset", "RP")
    LLE_P = part_2(phones_train_x, "Phones dataset", "MLLE")

    part_2_plot((PCA_F,ICA_F,RP_F,LLE_F,PCA_P,ICA_P,RP_P,LLE_P),"part_2.png")

    part_2_timer(fruits_train_x, phones_train_x)

if 1:
    # part_3 runner
    for dataset in (("Fruits",fruits_train_x,fruits_train_y),("Phones",phones_train_x,phones_train_y)):
        for method in ("KMeans", "EM"):
            for reduction in ("PCA", "ICA","RP","MLLE"):
                part_3(dataset,method,reduction)

if 1:
    # part_4 runner
    part_4(("Fruits",fruits_train_x,fruits_train_y),0.2)

if 1:
    # part_5 runner
    part_5(("Fruits",fruits_train_x,fruits_train_y),0.2)
print("---------")
print(time.time()-t," seconds to run whole assignment on 8 logical cores CPU")