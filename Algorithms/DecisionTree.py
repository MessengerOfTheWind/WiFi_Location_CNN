# Code adapted from: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

import math as m
import random as r

class DecisionTree:
    """Constructor for Decision Tree.

    Params:
    max_depth (int): determines tree depth
    min_size (int): if below the value, must stop splitting data at this point
    """
    def __init__(self,max_depth=1,min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size

    """Stores training data to be used by the model.

    Params:
    data (array): the data (X_train and Y_train merged)
    print_tree (bool): Conditional that prints tree if True
    """
    def fit(self, train_data, print_tree = False):
        self.train_data = train_data
        self.tree = self.build_tree()
        if print_tree:
            self.print_tree(self.tree)

    """Splits data  based on an attribute and an attribute value.

    Params:
    index (int): index we are currently looking at (attribute)
    value (int): discriminating value that determines which bin data ends up (attribute value)
    data (array): data that is being added to bins
    """
    def test_split(self, index, value, dataset):
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    """ Creates children from a given node or makes a terminal node.

    Params:
    node (dict): the tree node we are concerned with
    max_depth (int): the maximum depth a tree can reach
    min_size (int): the minimum number of items required before becoming terminal
    depth (int): current depth of tree
    """
    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # Check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return 
        # Check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # Process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth+1)
        # Process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth+1)
    
    """ Builds a decision tree.
    
    """
    def build_tree(self):
        root = self.get_split(self.train_data)
        self.split(root, self.max_depth, self.min_size, 1) # Initial depth is 1
        return root
    
    """ Makes a prediction using the decision tree.

    Params:
    node (dict): tree node we are considering
    row (array): data being used in prediction

    Returns:
    predict (func): a recursive function call, allows to navigate tree until we reach the solution
    terminal node (dict): returns the node that captures the data
    """
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']
    
    """ Splits a dataset into k folds.

    Params:
    n_folds (int): the number of folds we want

    Returns:
    dataset_split (array): dataset split into folds
    """
    def cross_validation_split(self, n_folds):
        dataset_split = []
        dataset_copy = self.train_data.copy()
        fold_size = int(len(self.train_data)/n_folds)
        for i in range(n_folds):
            fold = []
            while len(fold) < fold_size:
                index = r.randrange(len(dataset_copy))
                # Sampling with no replacement
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    """ Evaluates the decision tree performance.
    
    Params:
    n_folds (int): the number of folds we want to use

    Returns:
    scores (array): the score produced by each fold
    """
    def evaluate(self,n_folds): # Can modify to take in test set
        folds = self.cross_validation_split(n_folds)
        scores = []
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold) # Focus on the other folds
            train_set = sum(train_set, [])
            test_set = []
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.decision_tree(test_set)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    """ Decision Tree Algorithm.
    
    Params: 
    train (array): training data
    test (array): test data

    Returns:
    predictions (array): a list of predictions made by the decsion tree
    """
    def decision_tree(self, test):
        tree = self.tree
        predictions = []
        for row in test:
            prediction = self.predict(tree, row)
            predictions.append(prediction)
        return predictions
    
    """ Prints tree for visualisation of structure.

    Params:
    node (dict): Contains the tree information
    depth (int): current depth
    """
    def print_tree(self, node, depth=0):
        if isinstance(node,dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))

class DecisionTreeClassifier(DecisionTree):
    """ Constructor for DecisionTreeClassifier.

    Params:
    max_depth (int): determines tree depth
    min_size (int): if below this value, must stop splitting data at this point
    """
    def __init__(self,max_depth=1,min_size=1):
        super().__init__(max_depth,min_size)
    
    """Calculates the Gini index for a split.
    
    Params:
    groups (array): the data in each bin
    classes (array): the distinct class labels

    Returns:
    gini (float): Gini index for the split
    """
    def gini_index(self,groups,classes):
        # Count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # Avoid divide by 0
            if size == 0:
                continue
            score = 0.0
            # Score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini

    """Selects the best split for a given dataset

    Params:
    dataset: the data we are processing

    Returns:
    (dict): the variables describing the best split
    """
    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = m.inf, m.inf, m.inf, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                #print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value,'groups':b_groups}

    """ Creates a terminal tree node.

    Params:
    group (array): the bin we are considering

    Returns:
    (int): most common label
    """
    def to_terminal(self,group):
        labels = [row[-1] for row in group]
        return max(set(labels),key=labels.count)
    
    """ Calculates accuracy of algorithm.
    
    Params:
    actual (array): the actual label of the data
    predicted (array): the predicted label of the data

    Returns:
    (float): accuracy as a percentage
    """
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual))
    
class DecisionTreeRegressor(DecisionTree):
    """ Constructor for DecisionTreeRegressor.

    Params:
    max_depth (int): determines tree depth
    min_size (int): if below the value, must stop splitting data at this point
    """
    def __init__(self,max_depth=1,min_size=1):
        super().__init__(max_depth,min_size)
    
    """ Calculates Squared Error for DecisionTreeRegressor
    
    Params:
    groups (array): the distinct bins of the tree

    Returns:
    squared_error (float): the calculated squared error (rooted to make it smaller) 
    """
    def squared_error(self,groups):
        squared_error = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            y_hat = sum([row[-1] for row in group]) / size
            squared_error += sum([(row[-1] - y_hat)**2 for row in group])
        return squared_error**0.5
    
    """ Selects the best split for the given data.

    Params:
    dataset (array): the data we are considering
    
    Returns:
    (dict): a dictionary describing the ideal split
    """
    def get_split(self, dataset):
        b_index, b_value, b_score, b_groups = m.inf, m.inf, m.inf, None
        for index in range(len(dataset[0])-1): # Last position is label
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                squared_error = self.squared_error(groups)
                #print('X%d < %.3f Squared Error=%.3f' % ((index+1), row[index], squared_error))
                if squared_error < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], squared_error, groups
        return {'index':b_index,'value':b_value,'groups':b_groups}
    
    """ Creates a terminal node value for Regression Tree

    Params:
    group (array): the bin we are considering

    Returns:
    (float): average of the labels
    """
    def to_terminal(self, group):
        labels = [row[-1] for row in group]
        return sum(labels)/len(labels)
    
    """ Uses Mean Squared Error as accuracy metric for regression trees.
    
    Params:
    actual (array): the true labels for the data
    predicted (array): the predicted labels for the data

    Returns:
    (float): the calculated MSE 
    """
    def accuracy_metric(self, actual, predicted):
        total = 0
        for i in range(len(actual)):
            total += (actual[i] - predicted[i])**2
        return total / float(len(actual))
