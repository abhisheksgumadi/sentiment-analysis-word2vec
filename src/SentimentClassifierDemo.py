# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:58:16 2016

@author: abhishek
"""
import numpy as np
import matplotlib.pyplot as plt
import gensim
import logging 
import argparse
from SentimentClassifier import SentimentClassifier

from sklearn.metrics import roc_curve, auc, recall_score, precision_score, accuracy_score
from sklearn.cross_validation import train_test_split


def generate_feature_vectors(doc, model):
    """
    This method generates features for a document represented by a bag of words
    doc -- the document containing a bag of words seperated by space
    model -- the word2vec model
    
    returns the featue vector for the input document by averaging the word2vec 
    values of all the tokenized words in the input document
    """
    vec = np.zeros(300).reshape((1, 300))  
    count = 0
    for word in doc.split():
        if model.__contains__(word.strip()):
            count = count + 1
            vec += model[word.strip()]

    vec = vec / count
        
    return vec

def read_dataset(pos_dataset, neg_dataset):
    """
    This method reads the positive and negative dataset from the input file path
    pos_dataset -- the path to the file containing the positive samples
    neg_dataset -- the path to the file containing the negative samples
    
    returns a tuple containing both the positive and negative data
    """
    with open(pos_dataset, 'r') as pos_reviews:
            pos_reviews = pos_reviews.readlines()
            logging.info("all positive reviews are read")
        
    with open(neg_dataset, 'r') as neg_reviews:
            neg_reviews = neg_reviews.readlines()
            logging.info("all negative reviews are read")
        
    return (pos_reviews, neg_reviews)

def split_dataset_train_test(pos_dataset, neg_dataset, pred_values, test_prop):
    """
    This method splits a given dataset into train and test split. 
    pos_dataset -- The dataset representing the positive samples 
    neg_dataset -- The dataset representing the negative samples 
    test_prop -- The proportion of the input dataset that should be considered as the test data
    
    returns the split features for training and testing along with the class labels for both training and test evaluation
    """
    features_train, features_test, class_train, class_test = train_test_split(np.concatenate((pos_dataset, neg_dataset)), pred_values, test_size=test_prop)
    return (features_train, features_test, class_train, class_test)

def generate_features(word2vec_model, data):
    """
    This method generates features for each element of the data using the word2vec model
    word2vec_model -- The actual word2vec model
    data -- The raw data consisting of an array of textual elements, each of which will be input to the word2vec model
    
    returns the features encoded using word2vec for all elements of the input data
    """
    features = np.concatenate([generate_feature_vectors(s, model) for s in data])
    return features

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = """
    A sentiment analysis demo on movie reviews using word2vec""")
    
    parser.add_argument('-word2vec_model', type=str, required=True,
                        help="The path to the file containing the Google word2vec model (compressed .gz file)")
                        
    parser.add_argument('-positive_dataset', type=str, required=True,
                        help="The path to the file containing the positive dataset")

    parser.add_argument('-negative_dataset', type=str, required=True,
                        help="The path to the file containing the negative datset")
    
    parser.add_argument('-validation_proportion', type=float, required=True,
                        help="The fraction of the datset to be considerd for validation. Should be < 1.0 Ex: 0.2")
                        
    parser.add_argument('-log_file', type=str, required=True,
                        help="The path to the file that will contain the log")
    
    args = vars(parser.parse_args())
    
    print 'Arguments used: ' + str(args)
    
    word2vec_model_path = args['word2vec_model']
    pos_reviews_file = args['positive_dataset']
    neg_reviews_file = args['negative_dataset']
    valid_prop = args['validation_proportion']
    log_file = args['log_file']
    
    #get a new logger
    logger = logging.getLogger("SentimentClassification")
    #set the logging level
    logger.setLevel(logging.INFO)
    #set the log file name
    fh = logging.FileHandler(log_file)
    #set the formatter to print the time, name, logging level and the log message 
    formatter= logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #set the formatter
    fh.setFormatter(formatter)
    
    #add handler to the logger object
    logger.addHandler(fh)
    
    logger.info("program started")
    #load the pre-trained Google word2vec model
    model = gensim.models.Word2Vec.load_word2vec_format(word2vec_model_path, binary=True)
    logger.info("The word2vec model has been loaded")
        
    positive_reviews, negative_reviews = read_dataset(pos_reviews_file, neg_reviews_file)
    logger.info("dataset read")
    
    #create an array containing the values of the predicted variable for training
    y = np.concatenate((np.ones(len(positive_reviews)), np.zeros(len(negative_reviews))))
    
    #split the dataset into training and testing
    x_train, x_test, y_train, y_test = split_dataset_train_test(positive_reviews, negative_reviews, y, valid_prop)
    logger.info("The dataset has been split")

    #generate features for the training set    
    training_vectors = generate_features(model, x_train)
    logger.info("training features generated")
    
    #generate features from the testing set
    test_vectors = generate_features(model, x_test)
    logger.info("testing features generated")
    
    sc = SentimentClassifier()
    
    #train on the training set
    logger.info("Training process starts")    
    sc.fit(training_vectors, y_train)
    logger.info("Training process completes")
    
    #predict on the testing set
    logger.info("Testing process starts")
    pred_probs = sc.predict(test_vectors)
    logger.info("Testing process completes")
    
    #determine the class labels with a threshold of 0.5
    pred_class = [1 if ele > 0.5 else 0 for ele in pred_probs]
    
    recall = recall_score(y_test, pred_class)
    print "The recall score for the positive sentiment is %s percent" % str(recall*100.0)
    logger.info("Recall score is %s" % str(recall*100.0))
    
    precision = precision_score(y_test, pred_class)
    print "The precision score for the positive sentiment is %s percent" % str(precision*100.0)
    logger.info("Precision score is %s" % str(precision*100.0))
    
    accuracy = accuracy_score(y_test, pred_class)
    print "The accruacy score of the classifier is %s percent" % str(accuracy*100.0)
    logger.info("Accuracy score is %s" % str(accuracy*100.0))
    
    #generate a ROC curve and mention the area under curve to indicate the quality of the classifier
    logger.info("Generating the ROC curve")
    fpr,tpr,_ = roc_curve(y_test, pred_probs)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    logger.info("ROC curve generated")
    
    plt.show()
    
    