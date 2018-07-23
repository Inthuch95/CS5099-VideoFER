'''
Created on Jul 23, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.svm import LinearSVC 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from utils import load_au_single, plot_confusion_matrix
import pandas as pd
import numpy as np
import pickle
import os
import sys

# data_type = 'Basic'
data_type = 'Complex'
if data_type == 'Basic':
    DATA = pickle.load(open('../basic_emotions_data.pkl', 'rb'))
elif data_type == 'Complex':
    DATA = pickle.load(open('../complex_emotions_data.pkl', 'rb'))
else:
    print("Invalid data type")
    sys.exit()
EMOTIONS = DATA['EMOTIONS']
base_dir = '../SVM/' + data_type + '/AU/'

def train(model, X_train, y_train, kernel='linear'):
    model.fit(X_train, y_train)
    # save SVM model
    name = kernel + '.pkl'
    file = os.path.join(base_dir, name)
    with open(file, 'wb') as f:
        pickle.dump(model, f)
    return model

def evaluate_cv(model, X_train, y_train, kernel='linear'):
    # evaluate_vgg16 perfromance with 10-fold cv
    scores = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1)
    filename = kernel + '_cv.pkl'
    file = os.path.join(base_dir, filename)
    with open(file, 'wb') as f:
        pickle.dump(scores, f)

def display_score(scores):
    print('Scores: ', scores)
    print('Accuracy: %0.4f' % (scores.mean()))
    
def display_results(model, scores, kernel='linear'):
    display_score(scores)
    print('Test accuracy: %0.4f' % (model.score(X_test, y_test)))
    
    # plot confusion matrices
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)
    df.index.name = 'Actual'
    df.columns.name = 'Predicted'
    filename = 'confusion_matrix_' + kernel
    df.to_csv(base_dir+filename+'.csv')
    # complex emotions' cm is too large to plot
    if data_type == 'Basic':
        plt.subplots()
        plot_confusion_matrix(cm, title=kernel, class_names=EMOTIONS)
        plt.savefig(base_dir+'confusion_matrix_'+filename+'.png', format='png')

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_au_single(data_type=data_type)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
#     # evaluate_vgg16 linear svm
    print('Evaluating linear SVM')
    svm_linear = LinearSVC()
    evaluate_cv(svm_linear, X_train, y_train)
    svm_linear = train(svm_linear, X_train, y_train)
    
    svm_linear = pickle.load(open(base_dir+'linear.pkl', 'rb'))
    linear_scores = pickle.load(open(base_dir+'linear_cv.pkl', 'rb'))
    display_results(svm_linear, linear_scores)
    plt.show()