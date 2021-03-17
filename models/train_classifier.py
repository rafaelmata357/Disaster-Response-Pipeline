#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:    Rafael Mata M.
# DATE CREATED:  12 Marh 2021                                
# REVISED DATE:  12 March 2021
# PURPOSE: A ML Pipeline script to predict Categories form disastern messages and store in a file
#
#  Usage: python3 process_data.py messages_filepath  categories_filepath database_filename
#
#

# Imports python modules

import sys
import json
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#from sklearn.externals import joblib
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords


def load_data(database_filepath):
    ''' Function to read and load a database into a data frame 
    
        Params:
        -------
        database_filepath: str, file path of the database   

        Returns:
        --------
        df : pandas data frame
        
    '''

    engine = create_engine('sqlite:///'.foramt(database_filepath))
    df = pd.read_sql('SELECT * FROM MESSAGES', engine)

    returns df


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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