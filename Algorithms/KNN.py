from sklearn.metrics import r2_score

class KNN:
    """ Constructor method for my KNN implementation.

    Params:
    n_neighbours (int): the number of neigbours involved in producing the prediction
    """
    def __init__(self,n_neighbours = 1):
        self.n_neighbours = n_neighbours
        self.X_train = None
        self.Y_train = None

    """ Stores training data to be used by model. 
    
    Params:
    X_train (array): a collection of training samples
    Y_train (array): a collection of training labels
    """
    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    """ Produces the predictions for each test sample.

    Params:
    test_samples (array): a collection of test samples for prediction
    predictor (func): a function that predicts the label of the test sample

    Returns:
    predictions (array): An array of predictions
    """
    def predict(self,test_samples):
        predictions = []
        for test in test_samples:
            neighbours = self.get_neighbours(test)
            pred = self.predictor(neighbours)
            predictions.append(pred)
        return predictions

    """ Finds the distance between two points by calculating the euclidean distance between them.

    Params:
    x (array): an array of coordinates that represent position x
    y (array): an array of coordinates that represent position y

    Returns:
    (float): the euclidean distance between x and y.
    """
    def get_distance(self,x,y):
        return (sum([(y[i] - x[i])**2 for i in range(len(y))]))**0.5

    """ Finds the nearest neighbours to our test. 

    Params:
    test (array): an array of coordinates that represent test position.

    Returns:
    (array): the neighbours closest to our test point.
    (func): returns code for a single neighbour
    """
    def get_neighbours(self,test):
        if self.n_neighbours == 1:
            return self.get_neighbour(test)
        neighbours = []
        for i in range(len(self.X_train)):
            if not neighbours:
                neighbours.append((i, self.get_distance(self.X_train[i],test)))
            elif len(neighbours) < self.n_neighbours:
                dist = self.get_distance(self.X_train[i], test)
                for ptr in range(len(neighbours)):
                    if dist < neighbours[ptr][1]: # Compares distance to the ones in neighbours list
                        neighbours.insert(ptr,(i,dist)) # if distance lower inserts at that point
                        break # Exit the loop
            else:
                dist = self.get_distance(self.X_train[i], test)
                for ptr in range(len(neighbours)):
                    if dist < neighbours[ptr][1]: # Compares distance to the ones in neighbours list
                        neighbours.insert(ptr,(i,dist)) # if distance lower inserts at that point
                        del neighbours[-1] # Delete the furthest neighbour as we have greater neighbours
                        break # Exit the loop
        return neighbours
    
    """ Finds the single nearest neighbour to the test sample

    Params:
    test (array): an array of coordinates that represent test position.

    Returns:
    (array): the closest neighbour to the test point
    """
    def get_neighbour(self,test):
        neighbour = []
        for i in range(len(self.X_train)):
            if not neighbour: # Check if list is empty
                neighbour.append((i, self.get_distance(self.X_train[i],test)))
            else:
                dist = self.get_distance(self.X_train[i],test)
                if dist < neighbour[0][1]: # Compares the distances of the new neighbour and the stored nearest neighbour
                    neighbour[0] = (i, dist)
        return neighbour
            

class KNN_clf(KNN):
    """ Constructor for KNN classifier.

    Params:
    n_neighbours (int): the number of neigbours involved in producing the prediction
    """
    def __init__(self, n_neighbours=1):
        super().__init__(n_neighbours)

    """ Makes predictions based on the nearest neighbours.

    Params:
    neighbours (array): the set of nearest neighbours

    Returns:
    (int): the dominant label in the set of neighbours
    """
    def predictor(self, neighbours):
        count = {}
        # neighbour is a list storing [index of data item, distance]
        for neighbour in neighbours:
            if self.Y_train[neighbour[0]] in count:
                count[self.Y_train[neighbour[0]]] += 1
            else:
                count[self.Y_train[neighbour[0]]] = 1
        return max(count, key=count.get)
    
    """ Calculates the accuracy of the model, by counting the number of successful predictions

    Params:
    pred (array): predicted labels
    actual (array): actual labels

    Returns:
    (float): the percentage of successful predictions
    """
    def accuracy_metric(self, pred, actual):
        return sum(pred[i]==actual[i] for i in range(len(pred))) / len(pred)
    
class KNN_reg(KNN):
    """ Constructor for KNN regressor.

    Params:
    n_neighbours (int): the number of neigbours involved in producing the prediction
    """
    def __init__(self, n_neighbours=1, n_targets=1):
        super().__init__(n_neighbours)
        self.n_targets = n_targets

    """ Makes predictions based on the nearest neighbours.

    Params:
    neighbours (array): the set of nearest neighbours

    Returns:
    (func): multi_predictor if multiple targets
    (int): the average of the labels held by the set of neighbours
    """
    def predictor(self, neighbours):
        if self.n_targets > 1:
            return self.multi_predictor(neighbours)
        total = 0
        # neighbour is a list storing [index of data item, distance]
        for neighbour in neighbours:
            total += self.Y_train[neighbour[0]]
        return total/self.n_neighbours
    
    """ Makes predictions based on the nearest neighbours for multiple targets.

    Params:
    neighbours (array): the set of nearest neighbours

    Returns:
    (int): the average of the labels held by the set of neighbours
    """
    def multi_predictor(self, neighbours):
        targets = [0]*self.n_targets
        for neighbour in neighbours:
            for i in range(len(targets)):
                targets[i] += self.Y_train[neighbour[0]][i]
        return [targets[i]/self.n_neighbours for i in range(self.n_targets)]
        

    """ Calculates the perfomance of the regression model using MSE.

    Params:
    pred (array): predicted labels
    actual (array): actual labels

    Returns:
    (func): multi_mse if multiple targets
    (float): the calculated MSE
    """
    def MSE_metric(self, pred, actual):
        if self.n_targets > 1:
            return self.multi_mse(pred, actual)
        return sum([(actual[i] - pred[i])**2 for i in range(len(pred))]) / len(pred)
    
    """ Calculates the perfomance of the regression model using MSE for multiple target labels.

    Params:
    pred (array): predicted labels
    actual (array): actual labels

    Returns:
    (float): the calculated MSE
    """
    def multi_mse(self, pred, actual):
        total_error = 0
        for i in range(len(pred)):
            elemwise_loss = 0
            for j in range(self.n_targets):
                elemwise_loss += (actual[i][j] - pred[i][j])**2
            total_error += elemwise_loss/self.n_targets
        return total_error/len(pred)
    
    """ Calculates the r-squared score.

    Params:
    pred (array): predicted labels
    actual (array): actual labels

    Returns:
    (float): the r-squared score
    """
    def r2_metric(self, pred, actual):
        return r2_score(actual, pred)