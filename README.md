# sentiment-analysis-word2vec

## This project shows how to design a simple sentiment analysis for movie reviews using trained vectors from Google word2vec model

You can download the pre-trained Google word2vec vectors trained on Google news here - https://code.google.com/p/word2vec/

The required dataset is already present in this project

To know the list of parameters accepted by the program type

    python SentimentClassifierDemo.py -h
    
The following parameters are required to run the program

##### -word2vec_model  The path to the file containing the Google word2vec model (compressed .gz file)
##### -positive_dataset The path to the file containing the positive dataset. This file should contain one review per line 
##### -negative_dataset The path to the file containing the negative datset. This file should contain one review per line
##### -validation_proportion The fraction of the datset to be considerd for validation. Should be < 1.0 Ex: 0.2
##### -log_file LOG_FILE The path to the file that will contain the log messages
    


