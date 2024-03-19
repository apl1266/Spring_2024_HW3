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

warnings.filterwarnings("ignore")


def plotter(x,y,name,label,title,x_label="number of clusters",y_label="SSE"):
    for i in range(len(y)):
        plt.plot(x, y[i] , "-o", label=label[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(name)
    plt.clf()

def plotter_23(dat,file_name):
    plt.figure(figsize=(15,6))
    for j in range (6):
        plt.subplot(2,3, j+1)
        x, y, name, label, title, x_label, y_label=dat[j]
        for i in range(len(y)):
            plt.plot(x, y[i] , "-o", label=label[i])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()

def part_2_plot(data, file_name,label=("PCA","ICA","RP","MLLE")):
    plt.figure(figsize=(10,6))
    plt.subplot(2,2,1)
    for i in range(4):
        x,y=data[i]
        plt.plot(x, y, "-o", label=label[i])
    plt.xlabel("components number")
    plt.ylabel("reconstruction error")
    #plt.yscale("log")
    plt.title("Dimention reduction algorithms on fruits database")
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(4):
        x, y = data[i+4]
        plt.plot(x, y, "-o", label=label[i])
    plt.xlabel("components number")
    plt.ylabel("reconstruction error")
    #plt.yscale("log")
    plt.title("Dimention reduction algorithms on phones database")
    plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(4):
        x, y = data[i]
        plt.plot(x, y, "-o", label=label[i])
    plt.xlabel("components number")
    plt.ylabel("reconstruction error")
    plt.yscale("log")
    plt.title("Dimention reduction algorithms on fruits database")
    plt.legend()

    plt.subplot(2, 2, 4)
    for i in range(4):
        x, y = data[i + 4]
        plt.plot(x, y, "-o", label=label[i])
    plt.xlabel("components number")
    plt.ylabel("reconstruction error")
    plt.yscale("log")
    plt.title("Dimention reduction algorithms on phones database")
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_name)
    plt.clf()

def part_3_plotter(n_clus,SE,Homogeneity,Silhouette,dataset,method,reduction):
    if method=="KMeans":
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(n_clus, SE, "-o", label="SE")
        plt.xlabel("number of clusters")
        plt.ylabel("SSE")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(n_clus, Homogeneity, "-o", label="Homogeneity")
        plt.plot(n_clus, Silhouette, "-o", label="Silhouette")
        plt.xlabel("metric")
        plt.ylabel("SSE")
        plt.legend()
    else:
        plt.plot(n_clus, Homogeneity, "-o", label="Homogeneity")
        plt.plot(n_clus, Silhouette, "-o", label="Silhouette")
        plt.xlabel("metric")
        plt.ylabel("SSE")
        plt.legend()

    plt.tight_layout()

    plt.suptitle(method+" clustering "+reduction+" dim reduction on "+dataset+" dataset")
    plt.savefig("part_3_"+dataset+"_set_"+method+"_method_"+reduction+"_dim_reduction.png")
    plt.clf()

def part_4_plotter(x,train_score,test_score, dataset,method):
    plt.plot(x, train_score, "-o", label="Train sample")
    plt.plot(x, test_score, "-o", label="Validation sample")
    plt.xlabel("learning sample size")
    plt.ylabel("f score")
    plt.title(dataset+" "+method+" learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part_4_"+dataset+"_set_"+method+"_method_NN_learning_curve.png")
    plt.clf()

def part_5_plotter(x,train_score,test_score, dataset,reduction,method):
    plt.plot(x, train_score, "-o", label="Train sample")
    plt.plot(x, test_score, "-o", label="Validation sample")
    plt.xlabel("learning sample size")
    plt.ylabel("f score")
    plt.title(dataset+" "+reduction+" "+method+" learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part_5_"+dataset+"_set_"+reduction+"_"+method+"_method_NN_learning_curve.png")
    plt.clf()