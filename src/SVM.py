'''
Created on Jun 23, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.svm import LinearSVC 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from train_util import load_data_svm

def evaluate(model, X_train, y_train):
    # evaluate perfromance with 10-fold cv
    scores = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1)
    display_score(scores)

def display_score(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())

if __name__ == '__main__':
    X_train, y_train, _, _ = load_data_svm()
    # evaluate linear svm
    svm_linear = LinearSVC()
    evaluate(svm_linear, X_train, y_train)
    # evaluate svm with RBF kernel
    svm_rbf = SVC(kernel='rbf')
    evaluate(svm_rbf, X_train, y_train)