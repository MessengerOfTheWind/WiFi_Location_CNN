# Code adapted from https://towardsdatascience.com/implement-multiclass-svm-from-scratch-in-python-b141e43dc084

import numpy as np # for basic operations over arrays
from scipy.spatial import distance # to compute the Gaussian kernel
import cvxopt # to solve dual optimization problem
import copy # to copy numpy arrays

class SVM:
    linear = lambda x, xprime, c = 0: x @ xprime.T
    polynomial = lambda x, xprime, Q = 5: (1 + x @ xprime.T)**Q
    rbf = lambda x, xprime, gamma = 10: np.exp(- gamma * distance.cdist(x, xprime, 'sqeuclidean')) 
    kernel_funcs = {'linear': linear, 'polynomial': polynomial, 'rbf': rbf}

    """ Constructor for Support Vector Machine.

    Params:
    kernel (str): determines which kernel we use, can select linear, polynomial or rbf
    C (float): the regularization parameter of the SVM model
    k (int): the kernel parameter (is passed to the kernel function)
    """
    def __init__(self, kernel='rbf', C=1, gamma=5, k=2):
        # set the hyperparameters
        self.kernel_str = kernel
        self.kernel = SVM.kernel_funcs[kernel]
        self.gamma = gamma # hyperparameter
        self.C = C # regularization parameter
        self.k = k # number of classes
        
        # training data and support vectors
        self.X, self.y = None, None
        self.alphas = None
        
        # for multi-class classification
        self.multiclass = False
        self.clfs = []

    """ Fits the model to the data.

    Params:
    X (array): the training values
    y (array): the training labels
    eval_train (bool): conditional to determine whether we want to print training accuracy
    """
    def fit(self, X, y, eval_train=False):
        # if more than two unique labels, call the multiclass version
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y, eval_train)
        
        # if labels given in {0,1} change it to {-1,1}
        if set(np.unique(y)) == {0, 1}: y[y==0] = -1
        
        # ensure y is a Nx1 column vecotr (needed by CVXOPT)
        self.y = y.reshape(-1, 1).astype(np.double) # Has to be a column vector
        self.X = X
        N = X.shape[0] # Number of points
        
        # compute the kernel over all possible pairs of (x, x') in the data
        # by Numpy's vectorization this yields the matrix K
        self.K = self.kernel(X, X, self.gamma)
        
        ### Set up optimization parameters
        # For 1/2 x^T P x + q^T x
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N, 1)))
        
        # For Ax = b
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))
        
        # For Gx <= h
        G = cvxopt.matrix(np.vstack((-np.identity(N), np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N,1)), np.ones((N,1)) * self.C)))
        
        # Solve 
        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(sol['x']) # our solution
        
        # a Boolean array that flags points which are support vectors
        self.is_sv = ((self.alphas-1e-3 > 0)&(self.alphas <= self.C-1e-3)).squeeze() # alpha <= 1e-3 is approximately zero
        # an index of some margin support vector
        self.margin_sv = np.argmax((0 < self.alphas-1e-3)&(self.alphas < self.C-1e-3))
        
        if eval_train:
            print(f'Finished training with accuracy {self.evaluate(X,y)[0]} %')
    
    """ Produces predictions over given data.

    Params:
    X_t (array): the test samples we want to predict labels for

    Returns:
    np.sign(score) (array): test predictions (transfromed to be either 0 or 1)
    score (array): test predictions (raw) 
    """
    def predict(self, X_t):
        if self.multiclass: return self.multi_predict(X_t)
        # compute (x_s, y_s)
        x_s, y_s = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
        # find support vectors
        alphas, y, X= self.alphas[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]
        # compute the second term
        b = y_s - np.sum(alphas * y * self.kernel(X, x_s, self.k), axis=0)
        # compute the score
        score = np.sum(alphas * y * self.kernel(X, X_t, self.k), axis=0) + b
        return np.sign(score).astype(int), score
    
    """ Evaluates the accuracy of the predictions produced by the model.

    Params:
    X (array): test values
    y (array): test labels

    Returns:
    (float): the accuracy as a percentage 
    outputs (array): test predictions
    """
    def evaluate(self, X, y):
        if self.multiclass: return self.multi_evaluate(X,y)
        outputs, _ = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy*100, 2), outputs
    
    """Fits the SVM for multiple classes. (Use One vs Rest Classification)

    Params:
    X (array): the training data
    y (array): the training labels
    eval_train (bool): Conditional that determines whether we output the training accuracy
    """
    def multi_fit(self, X, y, eval_train=False):
        self.k = len(np.unique(y)) # Number of classes
        # for each pair of classes
        for i in range(self.k):
            # get the data for the pair
            Xs, Ys = X, copy.copy(y)
            # Change the labels to -1 and 1
            Ys[Ys!=i], Ys[Ys==i] = -1, +1
            #fit the classifier
            clf = SVM(kernel=self.kernel_str,C=self.C, gamma=self.gamma, k=self.k)
            clf.fit(Xs, Ys)
            # save the classifier
            self.clfs.append(clf)
        if eval_train:
            print(f'Finished training with accuracy {self.evaluate(X,y)[0]} %')
    
    """ Produces predictions over a given dataset (multi-classification).

    Params:
    X (array): test data

    Returns:
    np.argmax(preds, axis=1) (int): the dominant predicted label
    np.max(preds, axis=1) (int): the dominant predcted score
    """
    def multi_predict(self, X):
        # get the predictions from all classifiers
        N = X.shape[0]
        preds = np.zeros((N, self.k))
        for i, clf in enumerate(self.clfs):
            _, preds[:, i] = clf.predict(X)
        
        # get the argmax and the corresponding score
        return np.argmax(preds, axis=1), np.max(preds, axis=1)
    
    """ Evaluates multi-classification model.

    Params:
    X (array): test data
    y (array): true test labels
    
    Returns:
    (float): accuracy of the model as a percentage
    outputs (array): predicted labels for the test data
    """
    def multi_evaluate(self, X, y):
        outputs, _ = self.multi_predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy*100, 2), outputs