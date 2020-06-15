# -*- coding: utf-8 -*-
"""
Distance metric learning project for EEE4114F: ML
Test and evaluate differenct distance metrics for kNN.
Compare in Confusion matrix. 
Metrics to test:
    1. Euclidian 
    2. Manhattan 
    3. Mahalanobis 
    4. 'Optimal' as per Wang, Shijun & Jin, Rong. (2009). An Information Geometry Approach for Distance Metric Learning.. Journal of Machine Learning Research - Proceedings Track. 5. 591-598. 
    -> Extension: Might reduce 1 and 2 to general Minkowski distance, others e.g. cosine sim
Adapted from part1.ipynb assginment 
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy.stats import mode


def minkowskiDistance(x_a, x_b, p=2):
    """
    Calculates the minkowski distance between two vectors
    
    Arguments:
        x_a (np.array): shape [m_features, ] a single vector a
        x_b (np.array): shape [m_features, ] a single vector b
        p (int): Sets the Lp distance metric to use:
            1 - Manhattan
            2 - Euclidian 
            inf - Chebyshev
    
    Returns:
        distance (float): Minkowski distance between vectors x_a and x_b
    """
    
    distance = np.sum(np.abs(x_a - x_b)**p)**(1/p)
    return distance

def euclideanDistance(x_a, x_b):
    """
    Calculates the Euclidean distance between two vectors
    
    Arguments:
        x_a (np.array): shape [m_features, ] a single vector a
        x_b (np.array): shape [m_features, ] a single vector b
    
    Returns:
        distance (float): Euclidean distance between vectors x_a and x_b
    """
    
    return minkowskiDistance(x_a, x_b, 2)

def manhattanDistance(x_a, x_b):
    """
    Calculates the Manhattan distance between two vectors
    
    Arguments:
        x_a (np.array): shape [m_features, ] a single vector a
        x_b (np.array): shape [m_features, ] a single vector b
    
    Returns:
        distance (float): Manhattan distance between vectors x_a and x_b
    """
    return minkowskiDistance(x_a, x_b, 1)

def chebyshevDistance(x_a, x_b):
    """
    Calculates the Chebyshev distance between two vectors
    
    Arguments:
        x_a (np.array): shape [m_features, ] a single vector a
        x_b (np.array): shape [m_features, ] a single vector b
    
    Returns:
        distance (float): Chebyshev distance between vectors x_a and x_b
    """
    
    distance = np.max( np.abs(x_a - x_b) ) 
    return distance


def calculateDistances(x_test, X_in, distanceFunction):
    """
    Calculates the distance between a single test example, x_test,
    and a list of examples X_in. 
    
    Args:
        x_test (np.array): shape [n_features,] a single test example
        X_in (np.array): shape [n_samples, n_features] a list of examples to compare against.
    
    Returns:
        distance_list (list of float): The list containing the distances       
    """
    
    distance_list = []
    for example in X_in:
        distance_list.append(distanceFunction(example, x_test))
    return distance_list

def kNearestIndices(distance_list, k):
    """
    Determines the indices of the k nearest neighbours
    
    Arguments:
        distance_list (list of float): list of distances between a test point 
            and every training example
        k (int): the number of nearest neighbours to consider
    
    Returns:
        k_nearest_indices (array of int): shape [k,] array of the indices 
            corresponding to the k nearest neighbours
    """
    
    k_nearest_indices = np.array( np.argsort(distance_list)[:k] )
    return k_nearest_indices

def kNearestNeighbours(k_nearest_indices, X_in, Y_in):
    """
    Creates the dataset of k nearest neighbours
    
    Arguments:
        k_nearest_indices (array of int): shape [k,] array of the indices 
            corresponding to the k nearest neighbours
        X_in (np.array): shape [n_examples, n_features] the example data matrix to sample from
        Y_in (np.array): shape [n_examples, ] the label data matrix to sample from
    
    Returns:
        X_k (np.array): shape [k, n_features] the k nearest examples
        Y_k (np.array): shape [k, ] the labels corresponding to the k nearest examples
    """
    
    X_k = []
    Y_k = []

    for i in k_nearest_indices:
        X_k.append(X_in[i])
        Y_k.append(Y_in[i])
        
    X_k = np.array(X_k)
    Y_k = np.array(Y_k)
    return X_k, Y_k


def predict(x_test, X_in, Y_in, k, distanceFunction):
    """
    Predicts the class of a single test example
    
    Arguments:
        x_test (np.array): shape [n_features, ] the test example to classify
        X_in (np.array): shape [n_input_examples, n_features] the example data matrix to sample from
        Y_in (np.array): shape [n_input_labels, ] the label data matrix to sample from
    
    Returns:
        prediction (array): shape [1,] the number corresponding to the class 
    """
    
    distance_list = calculateDistances(x_test, X_in, distanceFunction)
    kNN_indices = kNearestIndices(distance_list, k)
    X_k, Y_k = kNearestNeighbours(kNN_indices, X_in, Y_in)
    prediction =  mode(Y_k, axis=None)[0]

    return prediction

def predictBatch(X_t, X_in, Y_in, k, distanceFunction):
    """
    Performs predictions over a batch of test examples
    
    Arguments:
        X_t (np.array): shape [n_test_examples, n_features]
        X_in (np.array): shape [n_input_examples, n_features]
        Y_in (np.array): shape [n_input_labels, ]
        k (int): number of nearest neighbours to consider
    
    Returns:
        predictions (np.array): shape [n_test_examples,] the array of predictions
        
    """
    predictions = []
    for x_t_i in X_t:
        predictions.append(predict(x_t_i, X_in, Y_in, k, distanceFunction)[0])
    
    return np.array(predictions)

def accuracy(Y_pred, Y_test):
    """
    Calculates the accuracy of the model 
    
    Arguments:
        Y_pred (np.array): shape [n_test_examples,] an array of model predictions
        Y_test (np.array): shape [n_test_labels,] an array of test labels to 
            evaluate the predictions against
    
    Returns:
        accuracy (float): the accuracy of the model
    """
    assert(Y_pred.shape == Y_test.shape)
    
    correct = 0
    total = len(Y_test)

    for i in range(total):
        if (Y_pred[i] == Y_test[i]):
            correct += 1
    
    accuracy = correct/total
    return accuracy

def run(X_train, X_test, Y_train, Y_test, k, distanceFunction=euclideanDistance):
    """
    Evaluates the model on the test data
    
    Arguments:
        X_train (np.array): shape [n_train_examples, n_features]
        X_test (np.array): shape [n_test_examples, n_features]
        Y_train (np.array): shape [n_train_examples, ]
        Y_test (np.array): shape [n_test_examples, ]
        k (int): number of nearest neighbours to consider
    
    Returns:
        test_accuracy (float): the final accuracy of your model 
    """
    Y_pred = predictBatch(X_test, X_train, Y_train, k, distanceFunction)
    test_accuracy = accuracy(Y_pred, Y_test)

    return test_accuracy

def main():
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X_iris = iris.data
    Y_iris = iris.target
    X_iris_train, X_iris_test, Y_iris_train, Y_iris_test = train_test_split(X_iris, Y_iris, test_size = 0.5)
    print( run(X_iris_train, X_iris_test, Y_iris_train, Y_iris_test, 4, chebyshevDistance) ) 
    
    
    
if __name__ == '__main__':
    main()