# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 20:19:34 2016

@author: abhishek
"""
import argparse
import gensim
import logging

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = """
    A procedure to build a Word2Vec model on a corpus of data""")
    
    parser.add_argument('-word2vec_model', type=str, required=True,
                        help="The path to the file that will contain the Word2Vec model")
                        
    parser.add_argument('-corpus', type=str, required=True,
                        help="The path to the file that contains the sentences, one per line")
    
    parser.add_argument('-log_file', type=str, required=True,
                        help="The path to the file that will contain the log")
                        
    args = vars(parser.parse_args())
    print 'Arguments used: ' + str(args)
    
    word2vec_model_path = args['word2vec_model']
    corpus_file         = args['corpus']
    log_file            = args['log_file']
    
    #get a new logger
    logger = logging.getLogger("Word2VecBuilder")
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
    f = open(corpus_file, 'r')
    
    lines = f.readlines()
    logger.info("data read")
    
    sentences = []
    
    for line in lines:
        doc = line.split()
        sentences.append(doc)
    
    logger.info("word2vec model building starts")
    model = gensim.models.Word2Vec(sentences, min_count=1, size=300)
    logger.info("word2vec model building completed")
    
    logger.info("word2vec model saving starts")
    model.save_word2vec_format(word2vec_model_path, binary=True)
    logger.info("word2vec model saving completed")