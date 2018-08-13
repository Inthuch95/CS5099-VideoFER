'''
Created on Aug 13, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.svm import LinearSVC 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from utils import load_vgg_single, load_au_single, plot_confusion_matrix, load_var
import pandas as pd
import numpy as np
import pickle
import os
import sys

class SVM():
    def __init__(self, model, base_dir, scores=[]):
        DATA = load_var(data_type)
        self.EMOTIONS = DATA['EMOTIONS']
        self.model = model
        self.scores = scores
        self.base_dir = '../SVM/' + data_type + '/' + feature + '/'

    def train(self, X_train, y_train, kernel='linear'):
        self.model.fit(X_train, y_train)
        # save SVM model
        name = kernel + '.pkl'
        file = os.path.join(self.base_dir, name)
        with open(file, 'wb') as f:
            pickle.dump(self.model, f)

    def evaluate_cv(self, X_train, y_train, kernel='linear'):
        # evaluate_vgg16 perfromance with 10-fold cv
        self.scores = cross_val_score(self.model, X_train, y_train, cv=10, n_jobs=-1)
        filename = kernel + '_cv.pkl'
        file = os.path.join(self.base_dir, filename)
        with open(file, 'wb') as f:
            pickle.dump(self.scores, f)
    
    def __display_score(self):
        print('CV scores: ', self.scores)
        print('CV accuracy: %0.4f' % (self.scores.mean()))
        
    def display_results(self, X_test, y_test, kernel='linear'):
        self.__display_score()
        print('Test accuracy: %0.4f' % (self.model.score(X_test, y_test)))
        
        # plot confusion matrices
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df = pd.DataFrame(cm, index=self.EMOTIONS, columns=self.EMOTIONS)
        df.index.name = 'Actual'
        df.columns.name = 'Predicted'
        filename = 'confusion_matrix_' + kernel
        df.to_csv(self.base_dir+filename+'.csv')
        plt.subplots()
        plot_confusion_matrix(cm, title=kernel, class_names=self.EMOTIONS)
        plt.savefig(self.base_dir+filename+'.png', format='png')
        plt.show()
    
if __name__ == '__main__':
    feature = 'VGG16'
    data_type = 'Basic'
    if feature == 'VGG16':
        X_train, y_train, X_test, y_test = load_vgg_single(data_type=data_type)
    elif feature == 'AU':
        X_train, y_train, X_test, y_test = load_au_single(data_type=data_type)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        print("Invalid feature")
        sys.exit()
    
    # Load existing SVM model
    base_dir = '../SVM/' + data_type + '/' + feature + '/'
    model = pickle.load(open(base_dir+'linear.pkl', 'rb'))
    scores = pickle.load(open(base_dir+'linear_cv.pkl', 'rb'))
    svm = SVM(model, base_dir, scores=scores)
    svm.display_results(X_test, y_test)
    
#     Train SVM from scratch
#     model = LinearSVC()
#     svm = SVM(model, base_dir)
#     svm.evaluate_cv(X_train, y_train)
#     svm.train(X_train, y_train)