#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:    Rafael Mata M.
# DATE CREATED:  9 Marh 2021                                
# REVISED DATE:  9 March 2021
# PURPOSE: A ETL script to process two datasets with Disaster and Categories messages and store in a database
#
#  Usage: python3 process_data.py messages_filepath  categories_filepath database_filename
#
#

# Imports python modules

import sys
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Function to load messages and categoris Disaster Datasets
    
       Parms:
       -------
       messages_filepath : str, messages dataset filepath
       categories_filepath : str, categories dataset filepath

       Returns:
       --------
       df : pandas DF, with both dataset merge
    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')  #Merge both datasets
    return df


def clean_data(df):
    '''Function to clean the merge dataset 
    
       Parms:
       -------
       df : pandas DF, with both dataset
   
       Returns:
       --------
       df : pandas DF, df with clean data
    
    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories.
    category_colnames = list(row.str.split('-',expand=True)[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-',expand=True)[1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    
    # drop the original categories column from `df`

    df.drop('categories', axis=1,inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):

    ''' Function to read a dataframe and store it in a database
    
       Parms:
       -------
       df : pandas DF,
       database_filename : str, database name
   
       Returns:
       --------
       None
    
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()