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
from scipy.spatial import distance as sci_distance
from sklearn.metrics import confusion_matrix

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

def mahalanobisDistance(x_a, x_b, iCov):
    """
    Calculates the Mahalanobis distance between two vectors
    
    Arguments:
        x_a (np.array): shape [m_features, ] a single vector a
        x_b (np.array): shape [m_features, ] a single vector b
        iCov (np.array): shape [m_features, m_features] inverse covarience Matrix of training data
    
    Returns:
        distance (float): Mahalanobis distance between vectors x_a and x_b
    """
    return sci_distance.mahalanobis(x_a, x_b, iCov)
    
def idealKernelMatrix(Y_in, l=0.5):
    """
    Calculates the "ideal kernel" as defined by equation (6) as per Wang et al (2009)
        
    Arguments:
        Y_in (np.array): shape [n_examples, ] the label data matrix to sample from. Encoded as 0->C to denote each class. 
        l (optional, float): Smoothing parameter. Default = 0.5
    
    Returns:
        KD^- (np.array): shape [n_examples, n_examples] the ideal kernel matrix
    """
    #Transfrom shape to that of paper:
    #Currently Y_in has dimension nx1 where n is the number of examples.
    #Currently in Y_in each element denotes which class the corresponing examples belongs to
    #i.e. 1 => belongs to class 2 
    #Will transform to from of cxn, where c is the number of classes 
    #Such that each column corresponds to an example and each row corresponds to a binary econding of the class
    Y_in_KD  = np.zeros(shape=(Y_in.shape[0], (max(Y_in)+1) ))

    for i in range(Y_in.shape[0]):
        Y_in_KD[i][Y_in[i]] = 1
    Y_in_KD = Y_in_KD.T #Swap nxc -> cxn
    return (Y_in_KD.T)@(Y_in_KD) + l*np.identity(Y_in_KD.shape[1])   

def optimalDistanceMetric(X_in, Y_in):
    """
    Calculates the 'optimal' distance metric as per Wang et al (2009) equation (9)
    
    Args:
        X_in (np.array): shape [n_examples, m_features] a list of examples to compare against.
        Y_in (np.array): shape [n_input_labels, ] the label data matrix to sample from

    Returns:
        A (np.array): shape [m_features, m_features] the 'optimal' distance matrix
    """
    X_in = X_in.T #In the paper they have m and n swapped. i.e. Each row is a feature and each column an example
    iKD = np.linalg.inv(idealKernelMatrix(Y_in))
    return np.linalg.inv( X_in@iKD@X_in.T )

def optimalDistance(x_a, x_b, A):
    """
    Calculates the 'optimal' distance between two vectors as per Wang et al (2009). 
    Where d(x,y) is calculated as per Xing et al (2003) equation (2). 
    
    Arguments:
        x_a (np.array): shape [m_features, ] a single vector a
        x_b (np.array): shape [m_features, ] a single vector b
        A (np.array): shape [m_features, m_features] the 'optimal' distance matrix
    
    Returns:
        distance (float): 'Optimal' distance between vectors x_a and x_b
    """
    return np.sqrt( (x_a - x_b).T@A@(x_a - x_b) )

def calculateDistances(x_test, X_in, Y_in, distanceFunction):
    """
    Calculates the distance between a single test example, x_test,
    and a list of examples X_in. 
    
    Args:
        x_test (np.array): shape [m_features,] a single test example
        X_in (np.array): shape [n_examples, m_features] a list of examples to compare against.
        Y_in (np.array): shape [n_input_labels, ] the label data matrix to sample from
        distanceFunction (function): The distance metric to use 

    Returns:
        distance_list (list of float): The list containing the distances       
    """
    distance_list = []
    if distanceFunction == optimalDistance:
        A = optimalDistanceMetric(X_in, Y_in)
        for example in X_in:
            distance_list.append(distanceFunction(example, x_test, A))
        
    elif distanceFunction == mahalanobisDistance:
        iCov = np.linalg.inv(np.cov(X_in, rowvar=False))
        for example in X_in:
            #TODO: SWAPPED EXAMPLE AND X_TEST
            distance_list.append(distanceFunction(example, x_test, iCov))
    else:
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
        X_in (np.array): shape [n_examples, m_features] the example data matrix to sample from
        Y_in (np.array): shape [n_examples, ] the label data matrix to sample from
    
    Returns:
        X_k (np.array): shape [k, m_features] the k nearest examples
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
        x_test (np.array): shape [m_features, ] the test example to classify
        X_in (np.array): shape [n_input_examples, n_features] the example data matrix to sample from
        Y_in (np.array): shape [n_input_labels, ] the label data matrix to sample from
        k (int): The number of nearest neighbours to consider
        distanceFunction (function): The distance metric to use 
        
    Returns:
        prediction (array): shape [1,] the number corresponding to the class 
    """
    
    distance_list = calculateDistances(x_test, X_in, Y_in, distanceFunction)
    kNN_indices = kNearestIndices(distance_list, k)
    X_k, Y_k = kNearestNeighbours(kNN_indices, X_in, Y_in)
    prediction =  mode(Y_k, axis=None)[0]

    return prediction

def predictBatch(X_t, X_in, Y_in, k, distanceFunction):
    """
    Performs predictions over a batch of test examples
    
    Arguments:
        X_t (np.array): shape [n_test_examples, m_features]
        X_in (np.array): shape [n_input_examples, m_features]
        Y_in (np.array): shape [n_input_labels, ]
        k (int): number of nearest neighbours to consider
        distanceFunction (function): The distance metric to use 

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
        distanceFunction (optional, function): The distance metric to use. DEFAULT is euclidean distance

    Returns:
        test_accuracy (float): the final accuracy of your model 
    """
    Y_pred = predictBatch(X_test, X_train, Y_train, k, distanceFunction)
    test_accuracy = accuracy(Y_pred, Y_test)
    # Example of a confusion matrix in Python

    return test_accuracy, confusion_matrix(Y_test, Y_pred, normalize='all')

def main():
    trials = 100
    import sklearn.datasets
    
    #Iris trials
    iris = sklearn.datasets.load_iris()
    X_iris = iris.data
    Y_iris = iris.target
    
    sum_manhat = 0
    con_man = 0
    
    sum_euclid = 0
    con_euclid = 0
    
    sum_cheby = 0
    con_cheby = 0
    
    sum_mahala= 0
    con_mahala = 0
    
    sum_opt = 0
    con_opt = 0
    for i in range(trials):
        X_iris_train, X_iris_test, Y_iris_train, Y_iris_test = train_test_split(X_iris, Y_iris, test_size = 0.5)
        
        temp1, temp2  = run(X_iris_train, X_iris_test, Y_iris_train, Y_iris_test, 4, manhattanDistance ) 
        sum_manhat += temp1
        con_man += temp2
        
        temp1, temp2  = run(X_iris_train, X_iris_test, Y_iris_train, Y_iris_test, 4, euclideanDistance) 
        sum_euclid += temp1
        con_euclid += temp2
        
        temp1, temp2  = run(X_iris_train, X_iris_test, Y_iris_train, Y_iris_test, 4, chebyshevDistance)
        sum_cheby += temp1
        con_cheby += temp2

        temp1, temp2  = run(X_iris_train, X_iris_test, Y_iris_train, Y_iris_test, 4, mahalanobisDistance) 
        sum_mahala += temp1
        con_mahala += temp2

        
        temp1, temp2  = run(X_iris_train, X_iris_test, Y_iris_train, Y_iris_test, 4, optimalDistance) 
        sum_opt += temp1
        con_opt += temp2
        
        
    print("----- IRIS -----")
    print("Manhattan accuracy:", sum_manhat/trials )
    print("Manhattan confusion:\n", con_man/trials, end='\n\n' )

    print("Euclidean accuracy:", sum_euclid/trials )
    print("Euclidean confusion:\n", con_euclid/trials, end='\n\n' )

    print("Chebyshev accuracy:", sum_cheby/trials )   
    print("Chebyshev confusion:\n", con_cheby/trials, end='\n\n' )

    print("Mahalanobis accuracy:", sum_mahala/trials )
    print("Mahalanobis confusion:\n", con_mahala/trials, end='\n\n' )
    
    print("Optimal accuracy:", sum_opt/trials )
    print("Optimal confusion:\n", con_opt/trials, end='\n\n' )

    #Wine trials
    wine = sklearn.datasets.load_wine()
    X_wine = wine.data
    Y_wine = wine.target
    
    
    sum_manhat = 0
    con_man = 0
    
    sum_euclid = 0
    con_euclid = 0
    
    sum_cheby = 0
    con_cheby = 0
    
    sum_mahala= 0
    con_mahala = 0
    
    sum_opt = 0
    con_opt = 0
    for i in range(trials):
        X_wine_train, X_wine_test, Y_wine_train, Y_wine_test = train_test_split(X_wine, Y_wine, test_size = 0.5)
        
        temp1, temp2  = run(X_wine_train, X_wine_test, Y_wine_train, Y_wine_test, 4, manhattanDistance ) 
        sum_manhat += temp1
        con_man += temp2
        
        temp1, temp2  = run(X_wine_train, X_wine_test, Y_wine_train, Y_wine_test, 4, euclideanDistance) 
        sum_euclid += temp1
        con_euclid += temp2
        
        temp1, temp2  = run(X_wine_train, X_wine_test, Y_wine_train, Y_wine_test, 4, chebyshevDistance)
        sum_cheby += temp1
        con_cheby += temp2

        temp1, temp2  = run(X_wine_train, X_wine_test, Y_wine_train, Y_wine_test, 4, mahalanobisDistance) 
        sum_mahala += temp1
        con_mahala += temp2

        
        temp1, temp2  = run(X_wine_train, X_wine_test, Y_wine_train, Y_wine_test, 4, optimalDistance) 
        sum_opt += temp1
        con_opt += temp2
        
        
    print("----- WINE -----")
    print("Manhattan accuracy:", sum_manhat/trials )
    print("Manhattan confusion:\n", con_man/trials, end='\n\n' )

    print("Euclidean accuracy:", sum_euclid/trials )
    print("Euclidean confusion:\n", con_euclid/trials, end='\n\n' )

    print("Chebyshev accuracy:", sum_cheby/trials )   
    print("Chebyshev confusion:\n", con_cheby/trials, end='\n\n' )

    print("Mahalanobis accuracy:", sum_mahala/trials )
    print("Mahalanobis confusion:\n", con_mahala/trials, end='\n\n' )
    
    print("Optimal accuracy:", sum_opt/trials )
    print("Optimal confusion:\n", con_opt/trials, end='\n\n' )
    
    
    #Breast Cancer trials
    cancer = sklearn.datasets.load_breast_cancer()
    X_cancer = cancer.data
    Y_cancer = cancer.target
    
    sum_manhat = 0
    con_man = 0
    
    sum_euclid = 0
    con_euclid = 0
    
    sum_cheby = 0
    con_cheby = 0
    
    sum_mahala= 0
    con_mahala = 0
    
    sum_opt = 0
    con_opt = 0
    for i in range(trials):
        X_cancer_train, X_cancer_test, Y_cancer_train, Y_cancer_test = train_test_split(X_cancer, Y_cancer, test_size = 0.5)
        
        temp1, temp2  = run(X_cancer_train, X_cancer_test, Y_cancer_train, Y_cancer_test, 4, manhattanDistance ) 
        sum_manhat += temp1
        con_man += temp2
        
        temp1, temp2  = run(X_cancer_train, X_cancer_test, Y_cancer_train, Y_cancer_test, 4, euclideanDistance) 
        sum_euclid += temp1
        con_euclid += temp2
        
        temp1, temp2  = run(X_cancer_train, X_cancer_test, Y_cancer_train, Y_cancer_test, 4, chebyshevDistance)
        sum_cheby += temp1
        con_cheby += temp2

        temp1, temp2  = run(X_cancer_train, X_cancer_test, Y_cancer_train, Y_cancer_test, 4, mahalanobisDistance) 
        sum_mahala += temp1
        con_mahala += temp2

        
        temp1, temp2  = run(X_cancer_train, X_cancer_test, Y_cancer_train, Y_cancer_test, 4, optimalDistance) 
        sum_opt += temp1
        con_opt += temp2
        
        
    print("----- Cancer -----")
    print("Manhattan accuracy:", sum_manhat/trials )
    print("Manhattan confusion:\n", con_man/trials, end='\n\n' )

    print("Euclidean accuracy:", sum_euclid/trials )
    print("Euclidean confusion:\n", con_euclid/trials, end='\n\n' )

    print("Chebyshev accuracy:", sum_cheby/trials )   
    print("Chebyshev confusion:\n", con_cheby/trials, end='\n\n' )

    print("Mahalanobis accuracy:", sum_mahala/trials )
    print("Mahalanobis confusion:\n", con_mahala/trials, end='\n\n' )
    
    print("Optimal accuracy:", sum_opt/trials )
    print("Optimal confusion:\n", con_opt/trials, end='\n\n' )
    
if __name__ == '__main__':
    main()