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

warnings.filterwarnings("ignore")




def part_1(train_x,train_y,name,method="KMeans"):
    n_max_clus = 30
    n_clus = []
    SE = []
    Silhouette=[]
    Homogeneity=[]
    t0=time.time()
    for i in range (2,n_max_clus):
        n_clus.append(i)
        if method=="KMeans":
            kmeans_clus = KMeans(n_clusters=i, max_iter=2000, random_state=812,n_init=10).fit(train_x)
            SE.append(kmeans_clus.inertia_)
            Homogeneity.append(homogeneity_score(train_y, kmeans_clus.labels_))
            Silhouette.append(silhouette_score(train_x, kmeans_clus.labels_, metric='euclidean'))
        else:
            kmeans_clus= GaussianMixture(n_components=i, max_iter=100, random_state=44,n_init=5).fit(train_x)
            Homogeneity.append(homogeneity_score(train_y, kmeans_clus.predict(train_x)))
            Silhouette.append(silhouette_score(train_x, kmeans_clus.predict(train_x), metric='euclidean'))

    #print(time.time()-t0," seconds to run "+method+" on "+name)

    if method == "KMeans":
        pass
    #    plotter(n_clus,(SE,),"part_1 "+name+"_"+method+".png",("Standard Square error for "+method+" for "+name,),method+" error "+name,"number of clusters","SSE")
    #plotter(n_clus, [Homogeneity,Silhouette], "part_1 "+name + "_"+method+"_S_H.png", ["Homogeneity "+method+" error " + name,"Silhouette "+method+" error " + name], method+" error " + name,"number of clusters", "Homogeneity, and Silhouette scores")
    return (n_clus,(SE,),"part_1 "+name+"_"+method+".png",("SSE for "+method+" for "+name,),method+" error "+name,"number of clusters","SSE"),(n_clus, [Homogeneity,Silhouette], "part_1 "+name + "_"+method+"_S_H.png", ["Homogeneity "+method+" error " + name,"Silhouette "+method+" error " + name], method+" error " + name,"number of clusters", "score")


def part_2(train_x,dataset,type_red):
    x=[]
    y=[]
    t0=time.time()
    n_max = 21
    if dataset=="Fruits dataset":
        n_max=35
    for i in range(2,n_max):
        x.append(i)
        if type_red=="PCA":
            pca = PCA(n_components=i, random_state=812)
        elif type_red=="ICA":
            pca = FastICA(n_components=i, random_state=812)
        elif type_red=="RP":
            pca=GaussianRandomProjection(n_components=i, random_state=812)
        else:
            pca=LLE(n_components=i, random_state=812)
        if not type_red=="LLE":
            x_red=pca.fit_transform(train_x)
            x_rec=pca.inverse_transform(x_red)
            y.append(mean_squared_error(train_x,x_rec))
        else:
            pca.fit_transform(train_x)
            y.append(pca.reconstruction_error_)
    return x,y

def part_2_timer(fruits,phones):
    print("Part_2 ------------")
    t=time.time()
    PCA(n_components=20, random_state=812).fit_transform(fruits)
    print(time.time()-t, " seconds to run PCA transformation on Fruits database")
    t = time.time()
    PCA(n_components=15, random_state=812).fit_transform(phones)
    print(time.time() - t, " seconds to run PCA transformation on Phones database")

    t = time.time()
    FastICA(n_components=20, random_state=812).fit_transform(fruits)
    print(time.time() - t, " seconds to run ICA transformation on Fruits database")
    t = time.time()
    FastICA(n_components=15, random_state=812).fit_transform(phones)
    print(time.time() - t, " seconds to run ICA transformation on Phones database")

    t = time.time()
    GaussianRandomProjection(n_components=29, random_state=812).fit_transform(fruits)
    print(time.time() - t, " seconds to run RP transformation on Fruits database")
    t = time.time()
    GaussianRandomProjection(n_components=18, random_state=812).fit_transform(phones)
    print(time.time() - t, " seconds to run RP transformation on Phones database")

    t = time.time()
    LLE(n_components=20, random_state=812).fit_transform(fruits)
    print(time.time() - t, " seconds to run LLE transformation on Fruits database")
    t = time.time()
    LLE(n_components=20, random_state=812).fit_transform(phones)
    print(time.time() - t, " seconds to run LLE transformation on Phones database")

def part_3(dataset,method,reduction):
    if (reduction=="PCA" or reduction=="ICA") and dataset[0]=="Fruits":
        n=20
    elif (reduction=="PCA" or reduction=="ICA") and dataset[0]=="Phones":
        n=15
    elif reduction=="RP" and dataset[0]=="Fruits":
        n=29
    elif reduction=="RP" and dataset[0]=="Phones":
        n=18
    else:
        n=20

    if reduction == "PCA":
        pca = PCA(n_components=n, random_state=812)
    elif reduction == "ICA":
        pca = FastICA(n_components=n, random_state=812)
    elif reduction == "RP":
        pca = GaussianRandomProjection(n_components=n, random_state=812)
    else:
        pca = LLE(n_components=n, random_state=812)

    t=time.time()
    train_x=pca.fit_transform(dataset[1])
    train_y=dataset[2]

    n_max_clus = 30
    n_clus = []
    SE = []
    Silhouette = []
    Homogeneity = []
    for i in range(2, n_max_clus):
        n_clus.append(i)
        if method == "KMeans":
            kmeans_clus = KMeans(n_clusters=i, max_iter=2000, random_state=812, n_init=10).fit(train_x)
            SE.append(kmeans_clus.inertia_)
            Homogeneity.append(homogeneity_score(train_y, kmeans_clus.labels_))
            Silhouette.append(silhouette_score(train_x, kmeans_clus.labels_, metric='euclidean'))
        else:
            kmeans_clus = GaussianMixture(n_components=i, max_iter=100, random_state=812, n_init=5).fit(train_x)
            Homogeneity.append(homogeneity_score(train_y, kmeans_clus.predict(train_x)))
            Silhouette.append(silhouette_score(train_x, kmeans_clus.predict(train_x), metric='euclidean'))
    print("Part 3")
    print(time.time()-t," seconds for part_3 "+dataset[0]+" dataset "+method+" method "+ reduction+" dimencity reduction method" )
    part_3_plotter(n_clus,SE,Homogeneity,Silhouette,dataset[0],method,reduction)

def part_4(dat,split):
    splitter=int(len(dat))-int(len(dat)*split//1)
    splitter=int((1-split)*np.shape(dat[1])[0])
    Train_x,Test_x,Train_y,Test_Y=dat[1][:splitter,:],dat[1][splitter:,:],dat[2].flatten()[:splitter],dat[2].flatten()[splitter:]
    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)

    a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier, Train_x, Train_y,train_sizes=np.linspace(0.25,1,20), scoring='f1_weighted',n_jobs=-1)
    part_4_plotter(len(Test_x)*np.linspace(0.25,1,20),train_score.mean(axis=1), test_score.mean(axis=1),dat[0],"")
    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    t=time.time()
    Classifier.fit(Train_x,Train_y)
    print(time.time()-t,"seconds to train vanilla NN on "+dat[0]+" dataset")
    print("Part 4 ------------------")
    print("-------")
    print(sklearn.metrics.classification_report(Test_Y, Classifier.predict(Test_x), digits=4))
    print("-------")

    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    pca=PCA(n_components=20, random_state=812).fit_transform(dat[1])
    Train_x, Test_x, Train_y, Test_Y = pca[:splitter, :], pca[splitter:, :], dat[2].flatten()[:splitter], dat[2].flatten()[splitter:]
    a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier, Train_x, Train_y,train_sizes=np.linspace(0.25, 1, 20),scoring='f1_weighted', n_jobs=-1)
    part_4_plotter(len(Test_x) * np.linspace(0.25, 1, 20), train_score.mean(axis=1), test_score.mean(axis=1), dat[0],"PCA")
    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    t = time.time()
    Classifier.fit(Train_x, Train_y)
    print("Part 4 ------------------")
    print(time.time() - t, "seconds to train PCA NN on " + dat[0] + " dataset")
    print("-------")
    print(sklearn.metrics.classification_report(Test_Y, Classifier.predict(Test_x), digits=4))
    print("-------")

    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    pca = FastICA(n_components=20, random_state=812).fit_transform(dat[1])
    Train_x, Test_x, Train_y, Test_Y = pca[:splitter, :], pca[splitter:, :], dat[2].flatten()[:splitter], dat[2].flatten()[splitter:]
    a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier, Train_x, Train_y,train_sizes=np.linspace(0.25, 1, 20),scoring='f1_weighted', n_jobs=-1)
    part_4_plotter(len(Test_x) * np.linspace(0.25, 1, 20), train_score.mean(axis=1), test_score.mean(axis=1), dat[0],"ICA")
    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    t = time.time()
    Classifier.fit(Train_x, Train_y)
    print("Part 4 ------------------")
    print(time.time() - t, "seconds to train ICA NN on " + dat[0] + " dataset")
    print("-------")
    print(sklearn.metrics.classification_report(Test_Y, Classifier.predict(Test_x), digits=4))
    print("-------")

    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    pca = GaussianRandomProjection(n_components=29, random_state=812).fit_transform(dat[1])
    Train_x, Test_x, Train_y, Test_Y = pca[:splitter, :], pca[splitter:, :], dat[2].flatten()[:splitter], dat[2].flatten()[splitter:]
    a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier, Train_x, Train_y,train_sizes=np.linspace(0.25, 1, 20),scoring='f1_weighted', n_jobs=-1)
    part_4_plotter(len(Test_x) * np.linspace(0.25, 1, 20), train_score.mean(axis=1), test_score.mean(axis=1), dat[0],"RP")
    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    t = time.time()
    Classifier.fit(Train_x, Train_y)
    print("Part 4 ------------------")
    print(time.time() - t, "seconds to train RP NN on " + dat[0] + " dataset")
    print("-------")
    print(sklearn.metrics.classification_report(Test_Y, Classifier.predict(Test_x), digits=4))
    print("-------")

    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    pca = LLE(n_components=20, random_state=812).fit_transform(dat[1])
    Train_x, Test_x, Train_y, Test_Y = pca[:splitter, :], pca[splitter:, :], dat[2].flatten()[:splitter], dat[2].flatten()[splitter:]
    a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier, Train_x, Train_y,train_sizes=np.linspace(0.25, 1, 20),scoring='f1_weighted', n_jobs=-1)
    part_4_plotter(len(Test_x) * np.linspace(0.25, 1, 20), train_score.mean(axis=1), test_score.mean(axis=1), dat[0],"LLE")
    Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu', learning_rate="constant",learning_rate_init=0.01, random_state=812)
    t = time.time()
    Classifier.fit(Train_x, Train_y)
    print("Part 4 ------------------")
    print(time.time() - t, "seconds to train LLE NN on " + dat[0] + " dataset")
    print("-------")
    print(sklearn.metrics.classification_report(Test_Y, Classifier.predict(Test_x), digits=4))
    print("-------")

def part_5(dat,split):
    splitter = int((1 - split) * np.shape(dat[1])[0])
    for reduction in ("Vanila","PCA","ICA","RP","LLE"):
        if reduction=="Vanila":
            pca=dat[1]
        if reduction=="PCA":
            pca = PCA(n_components=20, random_state=812).fit_transform(dat[1])
        if reduction == "ICA":
            pca = FastICA(n_components=20, random_state=812).fit_transform(dat[1])
        if reduction=="RP":
            pca = GaussianRandomProjection(n_components=29, random_state=812).fit_transform(dat[1])
        if reduction=="LLE":
            pca = LLE(n_components=20, random_state=812).fit_transform(dat[1])
        for method in ("KMeans","EM"):
            if method=="KMeans":
                kmeans_clus = KMeans(n_clusters=7, max_iter=2000, random_state=812, n_init=10).fit(pca)
                pca_labels = kmeans_clus.labels_
            else:
                kmeans_clus = GaussianMixture(n_components=7, max_iter=100, random_state=812, n_init=5).fit(pca)
                pca_labels=kmeans_clus.predict(pca)
            Train_x,Test_x,Train_y,Test_Y=pca[:splitter,:],pca[splitter:,:],pca_labels[:splitter],pca_labels[splitter:]
            Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu',learning_rate="constant", learning_rate_init=0.01, random_state=812)
            a, train_score, test_score = sklearn.model_selection.learning_curve(Classifier, Train_x, Train_y,train_sizes=np.linspace(0.25, 1, 20),scoring='f1_weighted', n_jobs=-1)
            part_5_plotter(len(Test_x) * np.linspace(0.25, 1, 20), train_score.mean(axis=1), test_score.mean(axis=1),dat[0], reduction,method)
            Classifier = MLPClassifier(tol=0.005, hidden_layer_sizes=[25, 25], activation='relu',learning_rate="constant", learning_rate_init=0.01, random_state=812)
            t = time.time()
            Classifier.fit(Train_x, Train_y)
            print("Part 5 -----------------")
            print(time.time() - t, "seconds to train "+reduction+" "+method+" NN on " + dat[0] + " dataset")
            print("-------")
            print(sklearn.metrics.classification_report(Test_Y, Classifier.predict(Test_x), digits=4))
            print("-------")