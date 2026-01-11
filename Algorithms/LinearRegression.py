# Code Adapted from CS5100 Data Analysis Lab 4

import numpy as np

class LinearRegression:
    """ Constructor for Linear Regression model.
    """
    def __init__(self):
        self.X_train = None
        self.Y_train = None
    
    """Stores training data to be used by the model.

    Params:
    X_train (array): the collection of training samples
    Y_train (array): the collection of training labels
    """
    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        self.xm, self.ym = self.matrices()

    """ Produces a data matrix.

    Returns:
    xm (array): a matrix of the training data
    ym (array): a matrix of the training labels
    """
    def matrices(self):
        xm, ym = [], []
        for n in range(len(self.X_train)):
            x, y = [1] + self.X_train[n].tolist(), self.Y_train[n]
            xm.append(x)
            ym.append(y)
        xm, ym = np.array(xm).T, np.array(ym).T
        return xm, ym
    
    """ Calculates Mean Squared Error.
    
    Params:
    actual (array): the true labels for the data
    predicted (array): the predicted labels for the data

    Returns:
    (float): the calculated MSE 
    """
    def MSE(self,actual,predicted):
        total = 0
        for i in range(len(actual)):
            total += (actual[i] - predicted[i])**2
        return total / float(len(actual))
    
    """ Makes predictions based on Linear Regression model

    Params:
    X_test (array): the collection of test samples
    Y_test (array): the collection of test labels

    Returns:
    pred_Train (array): predictions on train set
    pred_Test (array): predictions on test set
    MSEtrain (float): train MSE 
    MSEtest (float): test MSE
    """
    def predict(self, X_test, Y_test):
        # By construction lambdaLS = (XX.T)^-1 XY.T (so can find the global minimizer of RSS)
        q = np.linalg.pinv(self.xm @ self.xm.T)
        lambdaLS = (q @ (self.xm @ self.ym.T)).squeeze()
        
        pred_Train = np.array([lambdaLS @ ([1] + x.tolist()) for x in self.X_train])
        pred_Test = np.array([lambdaLS @ ([1] + x.tolist()) for x in X_test])
        actual_Train = self.Y_train.squeeze().tolist()
        actual_Test = Y_test.squeeze().tolist()
        
        MSEtrain = self.MSE(actual_Train, pred_Train)
        MSEtest = self.MSE(actual_Test, pred_Test)
        print('MSEtrain: ', MSEtrain)
        print('MSEtest: ', MSEtest)

        return pred_Train, pred_Test, MSEtrain, MSEtest
