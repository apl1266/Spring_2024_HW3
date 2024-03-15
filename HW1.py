import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import time

np.random.seed(812)

plots_11=False
second_validation=True

fruits_DT=0
phones_DT=0
fruits_boosting=0
phones_boosting=0
fruits_KNN=0
phones_KNN=0
fruits_NN=1
phones_NN=1
fruits_SVM=0
phones_SVM=0

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

def plot_11(train_f_score, validation_f_score, x_axis, title,x_label,file_name=False, extra_label=False):
    plt.plot(x_axis, train_f_score , "-o", label="train")
    plt.plot(x_axis, validation_f_score, "-o", label="validation")
    if extra_label:
        plt.plot([], [], ' ', label="Best fit parameters")
        plt.plot([], [], ' ', label=extra_label)
    plt.xlabel(x_label)
    plt.ylabel("f_score")
    plt.title(title)
    plt.legend()
    if file_name:
        plt.savefig(file_name)
    plt.clf()

def plot_12(x_1, y_1, x_axis_1, title_1,x_label_1,x_2, y_2, x_axis_2, title_2,x_label_2,file_name=False):
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.plot(x_axis_1, x_1 , "-o", label="train")
    plt.plot(x_axis_1, y_1, "-o", label="validation")
    plt.xlabel(x_label_1)
    plt.ylabel("f_score")
    plt.title(title_1)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x_axis_2, x_2, "-o", label="train")
    plt.plot(x_axis_2, y_2, "-o", label="validation")
    plt.xlabel(x_label_2)
    plt.ylabel("f_score")
    plt.title(title_2)
    plt.legend()
    if file_name:
        plt.savefig(file_name)
    plt.clf()

def lerning_best(x,y,Classifier,metric,search_param,curve_param,title,x_label,cv_num,file_name=False, verbose=False,x_test=False,y_test=False):
    findings=sklearn.model_selection.GridSearchCV(Classifier, search_param,scoring=metric,cv=cv_num,n_jobs=-1)
    findings.fit(x,y)
    if type(verbose)==type("Hi"):
        print("----------------------")
        print(verbose+" best parameters:")
        print(findings.best_params_)
        print("----------------------")
    if type(x_test)!=type(False) and type(verbose)==type("Hi"):
        print("----------------------")
        t0=time.time()
        DT=findings.best_estimator_.fit(x, y)
        t1=time.time()
        print(verbose + " time to train: ", t1-t0, " seconds")
        t0 = time.time()
        predictions = DT.predict(x_test)
        t1 = time.time()
        print(verbose + " time to test: ", t1 - t0, " seconds")
        print("----------------------")
        print(verbose + " test results:")
        print(sklearn.metrics.classification_report(y_test, predictions, digits=4))
        print(sklearn.metrics.confusion_matrix(y_test, predictions))
        print("----------------------")
    a, train_score, test_score = sklearn.model_selection.learning_curve(findings.best_estimator_, x,y, train_sizes=curve_param,scoring=metric, cv=cv_num, n_jobs=-1)
    if type(file_name)==type("Hi"):
        plot_11(train_score.mean(axis=1), test_score.mean(axis=1), curve_param * len(fruits_train_x), title,x_label, file_name)

    return train_score.mean(axis=1), test_score.mean(axis=1),findings.best_estimator_

def validation(x,y,Classifier,metric,curve_param_name,curve_param,title,x_label,cv_num,file_name=False):
    pass
    train_score, test_score = sklearn.model_selection.validation_curve(Classifier, x, y, param_name=curve_param_name,param_range=curve_param, scoring=metric, cv=cv_num, n_jobs=-1)
    if type(file_name)==type("Hi"):
        plot_11(train_score.mean(axis=1), test_score.mean(axis=1), curve_param, title,x_label, file_name)
    return train_score.mean(axis=1), test_score.mean(axis=1)


fruits_train_x,fruits_train_y,fruits_test_x,fruits_test_y= frames("fruits_data.csv",0.2)
phones_train_x,phones_train_y,phones_test_x,phones_test_y= frames("phones_data.csv",0.2)

learn_x="Number of trained samples"

if fruits_DT:
    Classifier=DecisionTreeClassifier()
    x=fruits_train_x
    y=fruits_train_y
    x_test=fruits_test_x
    y_test=fruits_test_y
    search_param = {
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': np.linspace(0.0,0.02,20)
    }
    curve_param=np.linspace(0.25,1,40)
    verbose = "Fruits DT"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="fruits_lerning_curve_DT.png"
    else:
        file_name=False

    x_1,y_1,Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    #Classifier = DecisionTreeClassifier(criterion="entropy")
    curve_param_name="ccp_alpha"
    curve_param_1=np.linspace(0.0,0.1,50)
    verbose = "Fruits DT"
    title_1 = "Validation curve "+verbose
    if plots_11:
        file_name = "fruits_validation_curve_DT.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, title_1, x_label_1,5,file_name)

    file_name = "fruits_DT.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)


if phones_DT:
    Classifier = DecisionTreeClassifier()
    x = phones_train_x
    y = phones_train_y
    x_test = phones_test_x
    y_test = phones_test_y
    search_param = {
        'criterion': ['gini', 'entropy'],
        'ccp_alpha': np.linspace(0.0, 0.05, 20)
    }
    curve_param = np.linspace(0.25, 1, 40)
    verbose = "Phones DT"
    title = "Lurning curve "+verbose
    if plots_11:
        file_name = "phones_lerning_curve_DT.png"
    else:
        file_name = False

    x_1,y_1, Classifier=lerning_best(x, y, Classifier, 'f1_weighted', search_param, curve_param, title,learn_x, 5, file_name, verbose, x_test, y_test)

    #Classifier = DecisionTreeClassifier(criterion="entropy")
    curve_param_name = "ccp_alpha"
    curve_param_1 = np.linspace(0.0, 0.1, 100)
    verbose = "Phones DT"
    title_1 = "Validation curve "+verbose
    if plots_11:
        file_name = "phones_validation_curve_DT.png"
    else:
        file_name = False
    x_label_1 = curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, title_1, x_label_1, 5, file_name)

    file_name = "phones_DT.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)

if fruits_boosting :
    Classifier=AdaBoostClassifier(DecisionTreeClassifier())
    x=fruits_train_x
    y=fruits_train_y
    x_test=fruits_test_x
    y_test=fruits_test_y
    search_param = {
    "estimator__splitter": ["best", "random"],
    "estimator__max_depth": np.linspace(1, 10, 10).astype(int),
    'n_estimators': np.linspace(10,50,6).astype(int),
    'estimator__criterion': ['entropy', 'gini']
}
    curve_param=np.linspace(0.25,1,20)
    verbose = "Fruits boosting"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="fruits_lerning_curve_boosting.png"
    else:
        file_name=False

    x_1,y_1, Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    #Classifier = DecisionTreeClassifier(criterion="entropy")
    curve_param_name='n_estimators'
    curve_param_1=np.linspace(10,100,11).astype(int)
    verbose = "Fruits boosting"
    title_1 = "Validation curve "+verbose
    if plots_11:
        file_name = "fruits_validation_curve_boosting.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, title_1, x_label_1,5,file_name)

    file_name = "fruits_boosting.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)

    if second_validation:
        curve_param_name = 'estimator__max_depth'
        curve_param = np.linspace(1, 10, 10).astype(int)
        verbose = "Fruits boosting"
        title = "Validation curve " + verbose
        if plots_11:
            file_name = "fruits_validation_curve_boosting_1.png"
        else:
            file_name = False
        x_label = curve_param_name
        x_1, y_1 = validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param, title_1, x_label_1, 5,
                              file_name)

        file_name = "fruits_boosting_1.png"
        plot_12(x_1, y_1, curve_param, title, curve_param_name, x_2, y_2, curve_param_1, title, x_label, file_name)

if phones_boosting :
    Classifier=AdaBoostClassifier(DecisionTreeClassifier())
    x = phones_train_x
    y = phones_train_y
    x_test = phones_test_x
    y_test = phones_test_y
    search_param = {
    "estimator__splitter": ["best", "random"],
    "estimator__max_depth": np.linspace(1, 10, 10).astype(int),
    'n_estimators': np.linspace(10,50,6).astype(int),
    'estimator__criterion': ['entropy', 'gini']
}
    curve_param=np.linspace(0.25,1,20)
    verbose = "Phones boosting"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="phones_lerning_curve_boosting.png"
    else:
        file_name=False

    x_1,y_1, Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    #Classifier = DecisionTreeClassifier(criterion="entropy")
    curve_param_name='n_estimators'
    curve_param_1=np.linspace(10,100,11).astype(int)
    verbose = "Phones boosting"
    title_1 = "Validation curve "+verbose
    if plots_11:
        file_name = "phones_validation_curve_boosting.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, title_1, x_label_1,5,file_name)

    file_name = "phones_boosting.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)

    if second_validation:
        curve_param_name = 'estimator__max_depth'
        curve_param = np.linspace(1, 10, 10).astype(int)
        verbose = "Phones boosting"
        title = "Validation curve " + verbose
        if plots_11:
            file_name = "phones_validation_curve_boosting_1.png"
        else:
            file_name = False
        x_label = curve_param_name
        x_1, y_1 = validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param, title, x_label, 5,
                              file_name)

        file_name = "phones_boosting_1.png"
        plot_12(x_1, y_1, curve_param, title, curve_param_name, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)


if fruits_KNN :
    Classifier=KNeighborsClassifier(n_jobs=-1)
    x=fruits_train_x
    y=fruits_train_y
    x_test=fruits_test_x
    y_test=fruits_test_y
    search_param = {
    "weights": ["uniform", "distance"],
    "algorithm":["ball_tree", "kd_tree", "brute"],
    "leaf_size": [1,2,3,4,5,6,7,8,9,10,20,30,40],
    "n_neighbors": np.linspace(1, 20, 20).astype(int),
}
    curve_param=np.linspace(0.25,1,20)
    verbose = "Fruits KNN"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="fruits_lerning_curve_KNN.png"
    else:
        file_name=False

    x_1,y_1, Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    curve_param_name="n_neighbors"
    curve_param_1=np.linspace(1, 20, 20).astype(int)
    verbose = "Fruits KNN"
    title_1 = "Validation curve "+verbose
    if plots_11:
        file_name = "fruits_validation_curve_KNN.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, title_1, x_label_1,5,file_name)

    file_name = "fruits_KNN.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)

if phones_KNN :
    Classifier=KNeighborsClassifier(n_jobs=-1)
    x = phones_train_x
    y = phones_train_y
    x_test = phones_test_x
    y_test = phones_test_y
    search_param = {
    "weights": ["uniform", "distance"],
    "algorithm":["ball_tree", "kd_tree", "brute"],
    "leaf_size": [1,2,3],
    "n_neighbors": np.linspace(1, 200, 21).astype(int),
}
    curve_param=np.linspace(0.25,1,20)
    verbose = "Phones KNN"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="phones_lerning_curve_KNN.png"
    else:
        file_name=False

    x_1,y_1, Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    curve_param_name="n_neighbors"
    curve_param_1=np.linspace(1, 500, 51).astype(int)
    verbose = "Phones KNN"
    title_1 = "Validation curve "+verbose
    if plots_11:
        file_name = "phones_validation_curve_KNN.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, title_1, x_label_1,5,file_name)

    file_name = "phones_KNN.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)

if fruits_NN :
    Classifier=MLPClassifier(tol=0.005)
    x=fruits_train_x
    y=fruits_train_y
    x_test=fruits_test_x
    y_test=fruits_test_y

    search_param = {
    "activation":["identity", "logistic", "tanh", "relu"],
    "hidden_layer_sizes": [[10,10],[15,15],[20,20],[25,25],[30,30],[40,40]],
    "learning_rate_init": [0.05,0.01, 0.02,0.5],
    "learning_rate": ['constant', "invscaling","adaptive"],
}

    curve_param=np.linspace(0.25,1,20)
    verbose = "Fruits NN"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="fruits_lerning_curve_NN.png"
    else:
        file_name=False

    x_1,y_1, Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    curve_param_name="n_iter_no_change"
    curve_param_1=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    verbose = "Fruits NN"
    title_1 = "Learning curve epochs "+verbose
    if plots_11:
        file_name = "fruits_learning_curve_NN_1.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, "learning curve fruits, epochs", x_label_1,5,file_name)

    file_name = "fruits_NN.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, "learning curve fruits, epochs", x_label_1, file_name)

    curve_param_name = "hidden_layer_sizes"
    curve_param_1 = [[10, 30], [20, 30], [30, 30], [40, 30], [50, 30], [60, 30], [70, 30],[80, 30],[90, 30], [100, 30]]
    verbose = "Fruits NN"
    title_1 = "Validation curve " + verbose
    if plots_11:
        file_name = "fruits_validation_curve_NN.png"
    else:
        file_name = False
    x_label_1 = curve_param_name
    x_2, y_2 = validation(x, y, Classifier, 'f1_weighted', curve_param_name, np.array(curve_param_1)[:, 0],"validation curve fruits, neurons per layer", x_label_1, 5, file_name)
    if second_validation:
        curve_param_name = "hidden_layer_sizes"
        curve_param = [[20, 10], [20, 20], [20, 30], [20, 40], [20, 50],[20, 60],[20, 70],[20, 80],[20, 90],[20, 100]]
        verbose = "Fruits NN"
        title = "Validation curve " + verbose
        if plots_11:
            file_name = "fruits_validation_curve_NN_1.png"
        else:
            file_name = False
        x_label = curve_param_name
        x_1, y_1 = validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param, "validation curve fruits, number of layer", x_label, 5,file_name)

        file_name = "fruits_NN_1.png"
        plot_12(x_1, y_1, np.array(curve_param)[:,1], "validation curve fruits, number of layers", x_label_1, x_2, y_2, np.array(curve_param_1)[:,0], "validation curve fruits, neurons per layer", x_label_1, file_name)


if phones_NN:
    Classifier=MLPClassifier(tol=0.005)
    x = phones_train_x
    y = phones_train_y
    x_test = phones_test_x
    y_test = phones_test_y

    search_param = {
    "activation":["identity", "logistic", "tanh", "relu"],
    "hidden_layer_sizes": [[10,10],[15,15],[20,20],[25,25],[30,30],[40,40]],
    "learning_rate_init": [0.05,0.01, 0.02,0.5],
    "learning_rate": ['constant', "invscaling","adaptive"],
}

    curve_param=np.linspace(0.25,1,20)
    verbose = "Phones NN"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="phones_lerning_curve_NN.png"
    else:
        file_name=False

    x_1,y_1, Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    curve_param_name = "n_iter_no_change"
    curve_param_1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    verbose = "Phones NN"
    title_1 = "Learning curve epochs"+verbose
    if plots_11:
        file_name = "phones_lurning_curve_NN_1.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, "learning curve phones, epochs", x_label_1,5,file_name)

    file_name = "phones_NN.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, "learning curve phones, epochs", x_label_1, file_name)

    curve_param_name = "hidden_layer_sizes"
    curve_param_1 = [[10, 30], [20, 30], [30, 30], [40, 30], [50, 30], [60, 30], [70, 30],[80, 30],[90, 30], [100, 30]]
    verbose = "Phones NN"
    title_1 = "Validation curve " + verbose
    if plots_11:
        file_name = "phones_validation_curve_NN.png"
    else:
        file_name = False
    x_label_1 = curve_param_name
    x_2, y_2 = validation(x, y, Classifier, 'f1_weighted', curve_param_name, np.array(curve_param_1)[:, 0],"validation curve phones, neurons per layer", x_label_1, 5, file_name)

    if second_validation:
        curve_param_name = "hidden_layer_sizes"
        curve_param = [[30, 10], [30, 20], [30, 30], [30, 40], [30, 50],[30, 60],[30, 70],[30, 80],[30, 90],[30, 100]]
        verbose = "Phones NN"
        title = "Validation curve " + verbose
        if plots_11:
            file_name = "phones_validation_curve_NN_1.png"
        else:
            file_name = False
        x_label = curve_param_name
        x_1, y_1 = validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param, "validation curve phones, number of layer", x_label, 5,file_name)

        file_name = "phones_NN_1.png"
        plot_12(x_1, y_1, np.array(curve_param)[:,1], "validation curve phones, number of layers", x_label_1, x_2, y_2, np.array(curve_param_1)[:,0], "validation curve phones, neurons per layer", x_label_1, file_name)


if fruits_SVM :
    Classifier=SVC()
    x = phones_train_x
    y = phones_train_y
    x_test = phones_test_x
    y_test = phones_test_y
    lena=x.shape[1]
    search_param = {
    'C': [0.2, 0.5, 1, 2],
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2, 3, 4, 5],
    'gamma': np.linspace(0.1,2,10)*lena
}
    curve_param=np.linspace(0.25,1,20)
    verbose = "Phones SVM"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="phones_lerning_curve_SVM.png"
    else:
        file_name=False

    x_1,y_1, Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    Classifier=SVC(kernel="poly")
    curve_param_name="degree"
    curve_param_1=[2, 3, 4, 5]
    verbose = "Phones SVM"
    title_1 = "Validation curve "+verbose
    if plots_11:
        file_name = "phones_validation_curve_SVM.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, title_1, x_label_1,5,file_name)

    file_name = "phones_SVM.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)


if fruits_SVM :
    Classifier=SVC()
    x=fruits_train_x
    y=fruits_train_y
    x_test=fruits_test_x
    y_test=fruits_test_y
    lena=x.shape[1]
    search_param = {
    'C': [0.2, 0.5, 1, 2],
    'kernel': ['linear', 'poly', 'rbf'],
    'degree': [2, 3, 4, 5],
    'gamma': np.linspace(0.1,2,10)*lena
}
    curve_param=np.linspace(0.25,1,20)
    verbose = "Fruits SVM"
    title="Lurning curve "+verbose
    if plots_11:
        file_name="fruits_lerning_curve_SVM.png"
    else:
        file_name=False

    x_1,y_1, Classifier=lerning_best(x,y,Classifier,'f1_weighted',search_param,curve_param,title,learn_x,5,file_name,verbose,x_test,y_test)

    Classifier=SVC(kernel="poly")
    curve_param_name="degree"
    curve_param_1=[2, 3, 4, 5]
    verbose = "Fruits SVM"
    title_1 = "Validation curve "+verbose
    if plots_11:
        file_name = "fruits_validation_curve_SVM.png"
    else:
        file_name = False
    x_label_1=curve_param_name
    x_2,y_2=validation(x, y, Classifier, 'f1_weighted', curve_param_name, curve_param_1, title_1, x_label_1,5,file_name)

    file_name = "fruits_SVM.png"
    plot_12(x_1, y_1, curve_param*len(x), title, learn_x, x_2, y_2, curve_param_1, title_1, x_label_1, file_name)