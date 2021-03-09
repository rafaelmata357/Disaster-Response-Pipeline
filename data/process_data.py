#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:    Rafael Mata M.
# DATE CREATED:  9 Marh 2021                                
# REVISED DATE:  9 March 2021
# PURPOSE: A ETL script to process two datasets with Disaster and Categories messages and store in a database
#      
#
#  Usage:
#         1. Create a new object with the class Benford, with an numpy array or list, or
#         2. Load a dataset with the method: load_dataset(args) or
#         3. Load an image with the method: load_image(args)
#         4. Analyze the data with the mehtod: apply_benford(args)
#         5. Plot the results or export to a file
#         6. Export the results to a file
#

# Imports python modules



import sys


def load_data(messages_filepath, categories_filepath):
    pass


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


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