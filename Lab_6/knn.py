import random

import numpy as np
import pandas as pd
from scipy import spatial
from scipy import stats

class KNN:


    """
    Implementation of the k-nearest neighbors algorithm for classification.
    """
    def __init__(self, k):
        """
        Takes one parameter.  k is the number of nearest neighbors to use
        to predict the output variable's value for a query point. 
        """
        self.k = k
        pass
        
    def fit(self, X, y):
        """
        Stores the reference points (X) and their known output values (y).
        """
        self.X = X
        self.y = y
        
    def predict_loop(self, X):
        """
        Predicts the output variable's values for the query points X using loops.
        """
        predicted = [] # Creates an empty list
        for i in range(len(X)): # F or loop that iterates for the length of X
            distances = [] # Creates an empty list for the distance values
            for y in range(len(self.X)): # Loop that iterates for the number of elements in self.X
                distances.append(spatial.distance.cdist([X[i]], [self.X[y]])[0]) # Uses spatial.distance.cdist in order to calculate the distance between a single row of X and a single row of self.X
            knn_matrix = pd.DataFrame() # Creates an empty pandas dataframe
            knn_matrix['distances'] = distances # Creates a distances column from the list of distances
            knn_matrix['labels'] = self.y # Creates a dataframe column that will have the labels for the associated disance measures
            sorted_matrix = knn_matrix.sort_values(by='distances') # Creates a new matrix where the distance values are sorted from smallest to largest according to distance
            predicted.append(stats.mode(sorted_matrix.labels.values[:self.k]).mode[0]) #Uses stats.mode function to find the most common classification of the first k nearest distances
        return np.array(predicted) #converts the list to a np.array to ensure consistent output as predict_numpy

    def predict_numpy(self, X):
        """
        Predicts the output variable's values for the query points X using numpy (no loops).
        """
        distances = spatial.distance.cdist(X, self.X) # Calculates the distances between X and self.X, the output being a matrix of size len(X), len(self.X)
        sorted_indexes = np.argsort(distances, axis=1).transpose()[:self.k].transpose() # Sorts the lowest computed distances by transversing accross 'distances', returning an array the size len(X), self.k
        targets = self.y[sorted_indexes] # Returns the associated target for the indexes that were sorted above in an array of size len(X), self.k
        return stats.mode(targets,axis=1).mode.flatten() # Returns a 1-D array of length X with the predictions for each row of X


