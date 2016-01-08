# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:58:16 2016

@author: abhishek
"""
import numpy as np
from sklearn.cross_validation import train_test_split
import gensim
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def generate_feature_vectors(doc, model, unknown_word_vector):
    
    vec = np.zeros(300).reshape((1, 300))  
    count = 0
    for word in doc.split():
        if model.__contains__(word.strip()):
            count = count + 1
            vec += model[word.strip()]

    vec = vec / count
        
    return vec


if __name__ == "__main__":
    
    model = gensim.models.Word2Vec.load_word2vec_format("/home/abhishek/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)    
    unknown_word_vector = np.ones(300)*0.1
    
    with open("/home/abhishek/chattermill/nlp-challenge/positive_reviews.txt", 'r') as positive_reviews:
        pos_reviews = positive_reviews.readlines()
        
    with open("/home/abhishek/chattermill/nlp-challenge/negative_reviews.txt", 'r') as negative_reviews:
        neg_reviews = negative_reviews.readlines()
        
    
    y = np.hstack((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
    
    training_vectors = np.concatenate([generate_feature_vectors(s, model, unknown_word_vector) for s in x_train])
    print len(x_train)
    print x_train[0]
    test_vectors = np.concatenate([generate_feature_vectors(s, model, unknown_word_vector) for s in x_test])
    
    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(training_vectors, y_train)
    
    print 'Test Accuracy: %.2f'%lr.score(test_vectors, y_test)
    
    pred_probas = lr.predict_proba(test_vectors)[:,1]

    fpr,tpr,_ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    
    plt.show()
    
    