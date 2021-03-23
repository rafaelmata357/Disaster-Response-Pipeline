#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:    Rafael Mata M.
# DATE CREATED:  12 Marh 2021                                
# REVISED DATE:  23 March 2021
# PURPOSE: A ML Pipeline script to predict Categories form disastern messages and store in a file
#
#  Usage: python3 train_classifier.py database_filename model_filepath
#
#

# import libraries

import json
import pandas as pd
import re
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import joblib


def load_data(database_filepath):
    ''' Function to read and load a database into a data frame 
    
        Params:
        -------
        database_filepath: str, file path of the database   

        Returns:
        --------
        df : pandas data frame
    '''

    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM MESSAGES', engine)
    X = df.message
    y = df.iloc[:,4:]
    category_names  = list(y.columns)
    
    return X, y, category_names



def tokenize(text):
    '''Fucntion to tokenize and clean the text
    
       Params:
       -------
       text : str, the text message

       Returns:
       -------
       clean_token: list, with the clean tokens
    '''

        
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
       
    
        clean_tokens.append(clean_tok)
  
    return clean_tokens


def build_model():
    '''Function to build the Model Pipeline
    
       Params:
       -------
       None

       Returns:
       ---------
       pipeline: sklearn pipeline model
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(max_df=0.5,max_features=5000,tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=200)))
    ])

    parameters = {'vect__max_df': (0.5, 0.75, 1.0),
              'vect__max_features': (None, 5000, 10000),
              'tfidf__use_idf': (True, False),
              'clf__estimator__n_estimators': [50, 100, 200],
              'clf__estimator__min_samples_split': [2, 3, 4]
             }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    ''' Function to evaluate the trainned model 
    
        Params:
        -------
        model: pipeline sklearn model
        X_test: numpy array, X test values
        Y_test: numpy array, Y labels
        category_names: list, category names

        Returns:
        ---------
        None
    
    '''
    
    # predict on test data
    y_pred = model.predict(X_test)
    print('accuracy {}'.format((y_pred == Y_test.values).mean()))

    for i in range(len(Y_test.values[0,:])):
        print('Target: {}'.format(category_names[i]))
        print('--'*30)
        print(classification_report(Y_test.values[:,i], y_pred[:,i]))

    return None


def save_model(model, model_filepath):

    ''' Function to save the model using joblib in pickle format 
    
        Params:
        ---------
        model: sklearn pipeline, trainned model
        model_filepath : str, file path

        Returns:
        ---------
        None
    '''
    
    #Save the file as pickle and compress

    joblib.dump(model, model_filepath, compress = ('bz2',9))

    return None

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()