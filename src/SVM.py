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
from utils import load_data_svm, plot_confusion_matrix
import numpy as np
import pickle
import os

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def train(model, X_train, y_train, kernel):
    model.fit(X_train, y_train)
    # save SVM model
    name = kernel + '.pkl'
    file = os.path.join('../SVM/', name)
    with open(file, 'wb') as f:
        pickle.dump(model, f)
    return model

def evaluate_cv(model, X_train, y_train, kernel='linear'):
    # evaluate perfromance with 10-fold cv
    scores = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1)
    filename = kernel + '_cv.pkl'
    file = os.path.join('../SVM/', filename)
    with open(file, 'wb') as f:
        pickle.dump(scores, f)

def display_score(scores):
    print('Scores: ', scores)
    print('Accuracy: %0.4f' % (scores.mean()))

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data_svm()
    # evaluate linear svm and rbf svm
    print('Evaluating linear SVM')
#     svm_linear = LinearSVC()
#     evaluate_cv(svm_linear, X_train, y_train, 'linear')
#     svm_linear = train(svm_linear, X_train, y_train, 'linear')
    
    svm_linear = pickle.load(open('../SVM/linear.pkl', 'rb'))
    linear_scores = pickle.load(open('../SVM/linear_cv.pkl', 'rb')) 
    display_score(linear_scores)
    print('Test accuracy: %0.4f' % (svm_linear.score(X_test, y_test)))
    y_pred = svm_linear.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm, title='Linear SVM', class_names=emotions)
    
    plt.show()