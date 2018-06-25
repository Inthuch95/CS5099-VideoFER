'''
Created on Jun 23, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.svm import LinearSVC 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from train_util import load_data_svm
import pickle
import os

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

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data_svm()
    # evaluate linear svm and rbf svm
#     svm_linear = LinearSVC()
#     evaluate_cv(svm_linear, X_train, y_train, 'linear')
#     svm_linear = train(svm_linear, X_train, y_train, 'linear')
#     print(svm_linear.score(X_test, y_test))
    
#     svm_rbf = SVC(kernel='rbf')
#     evaluate_cv(svm_rbf, X_train, y_train, 'rbf')
#     svm_rbf = train(svm_rbf, X_train, y_train, 'rbf')
#     print(svm_rbf.score(X_test, y_test))
    
    print('SVM with linear kernel')
    svm_linear = pickle.load(open('../SVM/linear.pkl', 'rb'))
    linear_scores = pickle.load(open('../SVM/linear_cv.pkl', 'rb')) 
    display_score(linear_scores)
    print('Test accuracy: ', svm_linear.score(X_test, y_test))
    
    print('')
    print('SVM with RBF kernel')
    svm_rbf = pickle.load(open('../SVM/rbf.pkl', 'rb'))
    rbf_scores = pickle.load(open('../SVM/rbf_cv.pkl', 'rb')) 
    display_score(rbf_scores)
    print('Test accuracy: ', svm_rbf.score(X_test, y_test))