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
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, homogeneity_score, silhouette_score
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.metrics import mean_squared_error
from sklearn.random_projection import GaussianRandomProjection
import os
import warnings

warnings.filterwarnings("ignore")

np.random.seed(812)


def frames(name, rat):
    pd_fruits = pd.read_csv(name)
    if name == "fruits_data.csv":
        pd_fruits.loc[pd_fruits["Class"] == "BERHI", "Class"] = 0
        pd_fruits.loc[pd_fruits["Class"] == "DEGLET", "Class"] = 1
        pd_fruits.loc[pd_fruits["Class"] == "DOKOL", "Class"] = 2
        pd_fruits.loc[pd_fruits["Class"] == "IRAQI", "Class"] = 3
        pd_fruits.loc[pd_fruits["Class"] == "ROTANA", "Class"] = 4
        pd_fruits.loc[pd_fruits["Class"] == "SAFAVI", "Class"] = 5
        pd_fruits.loc[pd_fruits["Class"] == "SOGAY", "Class"] = 6

    scaler = preprocessing.MinMaxScaler()
    np_fruits = pd_fruits.values
    np_fruits_scaled = scaler.fit_transform(np_fruits)
    np.random.shuffle(np_fruits_scaled)
    if name == "fruits_data.csv":
        np_fruits_scaled[:, -1] = np_fruits_scaled[:, -1] * 6
    else:
        np_fruits_scaled[:, -1] = np_fruits_scaled[:, -1] * 3
    split = int(np.shape(np_fruits_scaled)[0] * rat // 1)
    np_fruits_scaled[:, -1] = np_fruits_scaled[:, -1].astype(int)
    return np_fruits_scaled[split:, :-1], np_fruits_scaled[split:, -1].flatten(), np_fruits_scaled[:split,
                                                                                  :-1], np_fruits_scaled[:split,
                                                                                        -1].flatten()


def plotter(x, y, name, label, title, x_label="number of clusters", y_label="SSE"):
    for i in range(len(y)):
        plt.plot(x, y[i], "-o", label=label[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(name)
    plt.clf()


def plotter_32(dat, file_name):
    plt.figure(figsize=(15, 6))
    for j in range(6):
        plt.subplot(2, 3, j + 1)
        x, y, name, label, title, x_label, y_label = dat[j]
        for i in range(len(y)):
            plt.plot(x, y[i], "-o", label=label[i])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()


def plotter_scat(data, y, file_name, title):
    ax = plt.figure(figsize=(16, 10)).add_subplot(projection='3d')
    ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], c=y)
    ax.set_xlabel('axis 1')
    ax.set_ylabel('axis 2')
    ax.set_zlabel('axis 3')
    plt.title(title)
    plt.savefig(file_name)
    plt.clf()


def plotter_22_normal_scat(dat, file_name):
    plt.figure(figsize=(8, 6))
    for j in range(4):
        if j % 2 == 0:
            plt.subplot(2, 2, j + 1)
            x, y, name, label, title, x_label, y_label = dat[j]
            for i in range(len(y)):
                plt.plot(x, y[i], "-o", label=label[i])
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.legend()
        else:
            ax = plt.subplot(2, 2, j + 1, projection='3d')
            data, y, title = dat[j]
            ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], c=y, s=1)
            ax.set_xlabel('axis 1')
            ax.set_ylabel('axis 2')
            ax.set_zlabel('axis 3')
            ax.set_title(title)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()


def part_1(train_x, train_y, name, method="KMeans"):
    n_max_clus = 30
    n_clus = []
    SE = []
    Silhouette = []
    Homogeneity = []
    t0 = time.time()
    for i in range(2, n_max_clus):
        n_clus.append(i)
        if method == "KMeans":
            kmeans_clus = KMeans(n_clusters=i, max_iter=2000, random_state=812, n_init=10).fit(train_x)
            SE.append(kmeans_clus.inertia_)
            Homogeneity.append(homogeneity_score(train_y, kmeans_clus.labels_))
            Silhouette.append(silhouette_score(train_x, kmeans_clus.labels_, metric='euclidean'))
        else:
            kmeans_clus = GaussianMixture(n_components=i, max_iter=100, random_state=44, n_init=5).fit(train_x)
            Homogeneity.append(homogeneity_score(train_y, kmeans_clus.predict(train_x)))
            Silhouette.append(silhouette_score(train_x, kmeans_clus.predict(train_x), metric='euclidean'))

    # print(time.time()-t0," seconds to run "+method+" on "+name)

    if method == "KMeans":
        pass
    #    plotter(n_clus,(SE,),"part_1 "+name+"_"+method+".png",("Standard Square error for "+method+" for "+name,),method+" error "+name,"number of clusters","SSE")
    # plotter(n_clus, [Homogeneity,Silhouette], "part_1 "+name + "_"+method+"_S_H.png", ["Homogeneity "+method+" error " + name,"Silhouette "+method+" error " + name], method+" error " + name,"number of clusters", "Homogeneity, and Silhouette scores")
    return (n_clus, (SE,), "part_1 " + name + "_" + method + ".png", ("SSE for " + method + " for " + name,),
            method + " error " + name, "number of clusters", "SSE"), (
    n_clus, [Homogeneity, Silhouette], "part_1 " + name + "_" + method + "_S_H.png",
    ["Homogeneity " + method + " error " + name, "Silhouette " + method + " error " + name], method + " error " + name,
    "number of clusters", "score")


def part_2_PCA(train_x, dataset):
    x = []
    y = []
    t0 = time.time()
    n_max = 21
    if dataset == "Fruits dataset":
        n_max = 35
    for i in range(2, n_max):
        x.append(i)
        pca_transform = PCA(n_components=i, random_state=812)
        pca_transform.fit_transform(train_x)
        y.append(sum(pca_transform.explained_variance_ratio_))
    # print(time.time()-t0, " seconds for PCA on "+dataset)
    # plotter(x, (y,), "part_2 "+dataset + "_PCA_clustering.png", ("PCA_clustering for"+dataset,), "PCA_clustering for"+dataset, x_label="number of dimentions", y_label="total explained_variance_ratio")
    return (x, (y,), "part_2 " + dataset + "_PCA_clustering.png", ("PCA_clustering for" + dataset,),
            "PCA_clustering for" + dataset, "number of dimentions", "total explained_variance_ratio")


def part_2_LLE(train_x, dataset):
    x = []
    y = []
    t0 = time.time()
    for i in range(3, 30):
        x.append(i)
        LLE_f = LLE(n_neighbors=i, n_components=20, random_state=812)
        LLE_f.fit_transform(train_x)
        y.append(LLE_f.reconstruction_error_)
    print(time.time() - t0, " seconds for LLE on " + dataset)
    # plotter(x, (y,), "part_2 " + dataset + "_LLE_clustering.png", ("PCA_clustering for" + dataset,),"PCA_clustering for" + dataset, x_label="number of dimentions", y_label="total explained_variance_ratio")


def part_2(train_x, dataset, type_red):
    x = []
    y = []
    t0 = time.time()
    n_max = 21
    if dataset == "Fruits dataset":
        n_max = 35
    for i in range(2, n_max):
        x.append(i)
        if type_red == "PCA":
            pca = PCA(n_components=i, random_state=812)
        elif type_red == "ICA":
            pca = FastICA(n_components=i, random_state=812)
        elif type_red == "RP":
            pca = GaussianRandomProjection(n_components=i, random_state=812)
        else:
            pca = LLE(n_components=i, random_state=812)
        if not type_red == "LLE":
            x_red = pca.fit_transform(train_x)
            x_rec = pca.inverse_transform(x_red)
            y.append(mean_squared_error(train_x, x_rec))
        else:
            pca.fit_transform(train_x)
            y.append(pca.reconstruction_error_)
    return x, y


fruits_train_x, fruits_train_y, fruits_test_x, fruits_test_y = frames("fruits_data.csv", 0)
phones_train_x, phones_train_y, phones_test_x, phones_test_y = frames("phones_data.csv", 0)

if 0:
    # part_1 runner
    a_0, b_0 = part_1(fruits_train_x, fruits_train_y, "Fruits dataset", method="KMeans")
    a_1, b_1 = part_1(fruits_train_x, fruits_train_y, "Fruits dataset", method="EM")
    a_2, b_2 = part_1(phones_train_x, phones_train_y, "Phones dataset", method="KMeans")
    a_3, b_3 = part_1(phones_train_x, phones_train_y, "Phones dataset", method="EM")
    plotter_32((a_0, b_0, b_1, a_2, b_2, b_3), "part_1.png")
    t0 = time.time()
    KMeans(n_clusters=20, max_iter=2000, random_state=812, n_init=10).fit(fruits_train_x)
    print(time.time() - t0, " seconds to run KMEANS on FRUITS data set")
    t0 = time.time()
    KMeans(n_clusters=17, max_iter=2000, random_state=812, n_init=10).fit(phones_train_x)
    print(time.time() - t0, " seconds to run KMEANS on Phones data set")

    t0 = time.time()
    GaussianMixture(n_components=20, max_iter=100, random_state=44, n_init=5).fit(fruits_train_x)
    print(time.time() - t0, " seconds to run KMEANS on FRUITS data set")
    t0 = time.time()
    KMeans(n_clusters=17, max_iter=2000, random_state=812, n_init=10).fit(phones_train_x)
    print(time.time() - t0, " seconds to run KMEANS on Phones data set")

if 1:
    PCA_F = part_2(fruits_train_x, "Fruits dataset", "PCA")
    ICA_F = part_2(fruits_train_x, "Fruits dataset", "ICA")
    RP_F = part_2(fruits_train_x, "Fruits dataset", "RP")
    LLE_F = part_2(fruits_train_x, "Fruits dataset", "LLE")

    PCA_P = part_2(fruits_train_x, "Fruits dataset", "PCA")
    ICA_P = part_2(fruits_train_x, "Fruits dataset", "ICA")
    RP_P = part_2(fruits_train_x, "Fruits dataset", "RP")
    LLE_P = part_2(fruits_train_x, "Fruits dataset", "LLE")

if 0:
    # part_2 PCA runner
    a_0 = part_2_PCA(fruits_train_x, "Fruits dataset")
    a_1 = part_2_PCA(phones_train_x, "Phones dataset")
    t0 = time.time()
    pca_transform = PCA(n_components=20, random_state=812)
    x_0 = pca_transform.fit_transform(fruits_train_x)
    print(time.time() - t0, "seconds to run PCA on FRUITS dataset")
    b_0 = x_0, fruits_train_y, "PCA on Fruits data"

    t0 = time.time()
    pca_transform = PCA(n_components=17, random_state=812)
    x_1 = pca_transform.fit_transform(phones_train_x)
    print(time.time() - t0, "seconds to run PCA on PHONES dataset")
    b_1 = x_1, phones_train_y, "PCA on Phones data"
    plotter_22_normal_scat((a_0, b_0, a_1, b_1), "part_2_PCA.png")

if 0:
    part_2_LLE(fruits_train_x, "Fruits dataset")
    part_2_LLE(phones_train_x, "Phones dataset")

if 0:
    pca_transform = PCA(n_components=20, random_state=812)
    x = pca_transform.fit_transform(fruits_train_x)
    plotter_scat(x, fruits_train_y)

    pca_transform = PCA(n_components=17, random_state=812)
    x = pca_transform.fit_transform(phones_train_x)
    plotter_scat(x, phones_train_y)

if 0:
    # LLE creation
    if not os.path.exists("Fruits_LLE"):
        os.makedirs("Fruits_LLE")
    for i in range(2, 30):
        LLE_f = LLE(n_neighbors=i, n_components=20, random_state=812)
        x = LLE_f.fit_transform(fruits_train_x)
        plotter_scat(x, fruits_train_y, "Fruits_LLE/" + str(i) + ".png", "scater plot for " + str(i) + " neighbours")
    if not os.path.exists("Phones_LEE"):
        os.makedirs("Phones_LEE")
    for i in range(2, 30):
        LLE_f = LLE(n_neighbors=i, n_components=17, random_state=812)
        x = LLE_f.fit_transform(phones_train_x)
        plotter_scat(x, phones_train_y, "Phones_LEE/" + str(i) + ".png", "scater plot for " + str(i) + " neighbours")

if 1:
    # part_2 PCA runner
    pass
