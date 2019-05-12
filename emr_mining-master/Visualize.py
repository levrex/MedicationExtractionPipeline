# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 12:20:45 2018

@author: tdmaarseveen
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import itertools
from sklearn import metrics
from sklearn.model_selection import learning_curve
import pandas as pd

def circlePlot(freq_list, lbl_list, title, explode_list=None):
    """
    Params:
        freq_list = list with frequencies to plot
        lbl_list = list with labels
        title = title of the plot
    """
    plt.figure(figsize=[10, 10])
    
    plt.pie(
        freq_list,
        labels=['Typos', 'Correct'],
        shadow=False,
        explode=explode_list,
        startangle=60,
        autopct='%1.1f%%',
        )
    
    plt.axis('equal', labeldistance=-1.05)
    plt.title(title, fontsize='xx-large')
    plt.legend(freq_list, labels=lbl_list, loc="best", bbox_to_anchor=(0.3, 1., 0., 0.), fontsize='x-large')
    plt.tight_layout()
    return plt

def performancePlot(df, title):
    """
    Params:
        df = dataframe where the different values per key are the 
            performance characteristics
        title = title of the plot
    """
    plt.figure(figsize=(10, 10))
    colors = cm.rainbow(np.linspace(0, 1, 5))
    markers = ['o', 'v', '^', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    for xe, ye in zip(list(df.keys()),list(df.values())):
        ix = 0
        for i in ye:
            plt.scatter(xe, i, marker=markers[ix], s=100, edgecolors='black', linewidths=1, color=colors[ix])
            ix += 1
    
    legend_elements = [Line2D([0], [0], marker=markers[0], color=colors[0], label='Ppv', markersize=20, markeredgecolor='black'),
                       Line2D([0], [0], marker=markers[1], color=colors[1], label='Npv', markersize=20, markeredgecolor='black'),
                       Line2D([0], [0], marker=markers[2], color=colors[2], label='Sensitivity', markersize=20, markeredgecolor='black'),
                       Line2D([0], [0], marker=markers[3], color=colors[3], label='Specificity', markersize=20, markeredgecolor='black'),
                       Line2D([0], [0], marker=markers[4], color=colors[4], label='Accuracy', markersize=20, markeredgecolor='black')]       
    plt.legend(handles=legend_elements)      
    
    
    plt.axes().set_xticklabels(list(df.keys()), rotation='vertical') 
    plt.title(title)
    return plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Accepts probability scores as well as binary scores.
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt

def append_roc_curve(title, lbl, fpr, tpr, plt, clr='darkorange'):
    """
    Append ROC curve to plot
    
    title : title of the plot
    tpr : array consisting of true positive rate fractions
    fpr : array consisting of false positive rate fractions
    lbl : label
    plt : pre-existing plot
    pred : predictions made by the estimator
    """
    roc_auc = np.trapz(tpr,fpr)
    lw = 2
    plt.plot(fpr, tpr, color=clr,
             lw=lw, label=lbl + ' (AUC = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Sensitiviteit (TPR)')
    plt.xlabel('1 - Specificiteit (FPR)')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    return plt

def plot_roc_curve(estimator, title, X, y):
    """
    Generate an ROC plot
    
    estimator : predicting model
    title : title of the plot
    X : array-like -> Test vector
    y : array-like vector consisting of all the actual classes
    y_b : binarized classes (y -> 1 and n -> 0)
    pred : predictions made by the estimator
    """
    pred = estimator.predict_proba(X)[:,1] 
    y_b = y.copy()
    for i in range(len(y)): # MAKE BINARY (y = 1, n = 0)
        y_b[i] = int(y_b[i] == 'y')
    fpr, tpr, threshold = metrics.roc_curve(list(y_b), list(pred), pos_label=1)
    auc = np.trapz(tpr,fpr)
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(threshold, index = i)})
    plt.figure(figsize=(16,6))
    ax1 = plt.subplot(131)
    ax1.set_title(title)
    ax1.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % auc)
    ax1.legend(loc = 'lower right')
    ax1.plot([0, 1], [0, 1],'r--')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Sensitivity (TPR)')
    ax1.set_xlabel('1 - Specificity (FPR)')
    
    ax2, cut_off = plot_cutoff_roc(roc, title)
    plt.tight_layout()
    print(roc.iloc[(roc.tf-0).abs().argsort()[:1]])
    return plt, cut_off

def plot_multi_roc(models, title, lbls, X_train, X_test, y_train, y_test):
    """ 
    models = list of Pipelines (sklearn)
    """
    colors = ['darkblue', 'blue', 'cyan', 'lightblue', 'wheat', 'green']
    for x in range(len(models)):
        estimator = models[x].fit(X_train, y_train)
        pred = estimator.predict_proba(X_test)[:,1]
        y_b = y_test.copy()
        for i in range(len(y_test)): # MAKE BINARY (y = 1, n = 0)
            y_b[i] = int(y_b[i] == 'y')
        fpr, tpr, threshold = metrics.roc_curve(list(y_b), list(pred), pos_label=1)
        auc = np.trapz(tpr,fpr)
        i = np.arange(len(tpr)) # index for df
        plt.plot(fpr, tpr, 'xkcd:' + colors[x], label = lbls[x] + ' (AUC = %0.2f)' % auc)
    #plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.ylabel('Sensitivity (TPR)')
    plt.xlabel('1 - Specificity (FPR)')
    plt.title(title)
    plt.rcParams.update({'font.size': 12})
    return plt

def plot_cutoff_roc(roc, title):
    """
    Generate an ROC Cut off plot
    
    roc = dataframe containing all the relevant info for the ROC curve
    axs[1] features subplot
    """
    ax2 = plt.subplot(132)
    cut_off = roc.iloc[(roc.tf-0).abs().argsort()[:1]]['thresholds'].values[0]
    ax2.plot(roc['tpr'], label = 'Cut off = %0.2f' % cut_off)
    ax2.legend(loc = 'lower right')
    ax2.plot(roc['1-fpr'], color = 'red')
    ax2.set_xlabel('Datapoints')
    ax2.set_ylabel('True Positive Rate', color = 'blue')
    ax2a = ax2.twinx()
    ax2a.set_ylabel('1 - False Positive Rate', color = 'red')
    ax2.set_title(title + ' - Cut off')
    ax2.set_ylim([0, 1])
    return ax2, cut_off

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
        
    Code from sklearn tutorial 
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt