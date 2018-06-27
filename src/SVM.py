'''
Created on Jun 23, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.svm import LinearSVC 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from train_util import load_data_svm
import numpy as np
import pandas as pd
import pickle
import os

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def train(model, X_train, y_train, kernel):
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    # save SVM model
    name = kernel + '.pkl'
    file = os.path.join('../SVM/', name)
    with open(file, 'wb') as f:
        pickle.dump(model, f)
    return model

def evaluate_cv(model, X_train, y_train, kernel):
    # evaluate perfromance with 10-fold cv
    scores = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1)
    display_score(scores)
    filename = kernel + '_cv.pkl'
    file = os.path.join('../SVM/', filename)
    with open(file, 'wb') as f:
        pickle.dump(scores, f)

def display_score(scores):
    print('Scores: ', scores)
    print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Reds, class_names=None):
    # create confusion matrix plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels(class_names)
    plt.yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.3f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data_svm()
    # evaluate linear svm and rbf svm
#     print('Evaluating linear SVM')
#     svm_linear = LinearSVC()
#     evaluate_cv(svm_linear, X_train, y_train, 'linear')
#     svm_linear = train(svm_linear, X_train, y_train, 'linear')
#     print(svm_linear.score(X_test, y_test))
#     
#     print('Evaluating rbf kernel SVM')
#     svm_rbf = SVC(kernel='rbf')
#     evaluate_cv(svm_rbf, X_train, y_train, 'rbf')
#     svm_rbf = train(svm_rbf, X_train, y_train, 'rbf')
#     print(svm_rbf.score(X_test, y_test))
    
    print('SVM with linear kernel')
    svm_linear = pickle.load(open('../SVM/linear.pkl', 'rb'))
    linear_scores = pickle.load(open('../SVM/linear_cv.pkl', 'rb')) 
    display_score(linear_scores)
    print('Test accuracy: ', svm_linear.score(X_test, y_test))
    y_pred = svm_linear.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm, title='Linear', class_names=emotions)
     
    print('')
    print('SVM with RBF kernel')
    svm_rbf = pickle.load(open('../SVM/rbf.pkl', 'rb'))
    rbf_scores = pickle.load(open('../SVM/rbf_cv.pkl', 'rb')) 
    display_score(rbf_scores)
    print('Test accuracy: ', svm_rbf.score(X_test, y_test))
    y_pred = svm_rbf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(cm, title='RBF', class_names=emotions)
    
    plt.show()