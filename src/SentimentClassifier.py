# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:58:16 2016

@author: abhishek
"""
import logging 

from sklearn.linear_model import LogisticRegression

class SentimentClassifier(object):
    """
    This class uses the Scikit Learn Logistic Regression classifier to train a sentiment classifier
    """
    
    def __init__(self):
        """
        This method initializes an instance of the SentimentClassifier class and instantiates the machine learning classifier
        """
        self.lr = LogisticRegression(class_weight='auto', penalty='l2')
        module_logger = logging.getLogger("SentimentClassification.classifier")
        module_logger.info("SentimentClassifier instantiated")
            
    def fit(self, x_train, y):
        """
        This method fits the input feature set of the training data
        x_train -- an array containing the feature vector for all elements of the training data
        y -- an array representing the class labels for the training elements
        """
        module_logger = logging.getLogger("SentimentClassification.fit")
        module_logger.info("training phase started")
        self.lr.fit(x_train, y)
        module_logger.info("training phase complete")
        
    def predict(self, x_test):
        """
        This method predicts the class for each of the feature vectors in the input 
        x_test -- an array of feature vectors used for prediction
        
        returns the predicted probabailities for each of the feature vector
        """
        module_logger = logging.getLogger("SentimentClassification.predict")
        module_logger.info("prediction phase started")
        pred_probas = self.lr.predict_proba(x_test)[:,1]
        module_logger.info("prediction phase completed")
        return pred_probas
        
       