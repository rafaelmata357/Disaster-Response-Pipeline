# Disaster Response Pipeline

## Project Motivation

As part of the Udacity Data Science nanodegree this project analyzes disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages


## Installation:

Clone the repository to the local machine

`$ gh repo clone rafaelmata357/Disaster-Response-Pipeline`

These libraries are used

- pandas
- numpy 
- flask
- nltk
- json
- sklearn
- flask
- sqlalchemy
- re
- sys
- joblib


The python version used: **3.8**

## Files in the repository

- README.md : This file
- disaster_categories.csv : CSV file containning the different messages categories
- disaster_messages.csv : CSV file containning the different disaster messages
- DisasterMessages.db : SQL Database with the messages and their categories
- process_data.py : ETL Python script to load, process, clean and store in a database the messages and categories
- train_classifier.py : Machine learning pipeline to train and classify the messages
- run.py: Python script to run the web app
- go.html : HTML file
- master.html : HTML file
- fig1,fig2, fig3 : jpeg files containing screen shots with output examples
- messages_db.db : SQLite with the messages
- model.pkl : pickle file with the optimize model


## Instructions to run the app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## How To Interact With the Project 

- The project in this repository follows the CRISPM methodology:

1. Business Understanding

    - Following a disaster millions of messages are generated via direct or via social media, and sent  asking for help or support during the events when the organizations have a reduce capacity to filter, process and attend the most important ones, classifying in different categories the messages (water, energy, infrastructure, etc) to be attended by the right organization, this process could be optimized using Machine learning to classify the messages.

2. Data Understanding
    - The dataset are pre-labeled messages in 36 categories, there are two datasets:
        - messages: a dataset containing 26248 messages from different sources: direct, social media or news 
        - categories : label data for each messages containing 36 posible categories
       
    - Different descriptive statistics are used for EDA and data visualization:
        - Histogram to analyze the different messages genres ![alt text](https://github.com/rafaelmata357/Disaster-Response-Pipeline/blob/main/Fig%201.png 'Messages distribution by genre')
        - Histogram to analyze the messages classification by category ![alt text](https://github.com/rafaelmata357/Disaster-Response-Pipeline/blob/main/Fig%202.png 'Messages distribution by category label')

    From the data above is clear that there is a data unbalance, the related, aid-related, weeather-related and direct concetrate most of the messages categories, it could be a good idea to increase the dataset messages to balance the different messages categories and have a better model.
       

3. Data preparation
    - An ETL pipeline is used:
        - Read the datasets
        - Merge the datasets
        - Split the categories
        - Convert the categories to numbers (0 or 1)
        - Remove the duplicates
        - Save the clean dataset into a SQLite database

4. Data Modeling
    - A Machile learning pipeline is used, containing:
        - Vectorizer, using a tokenizer
            - normalize text
            - word tokenize
            - remove stop words
            - lemmatize
        - TfidfTransformer
        - MultiOutputClassifier
           - RandomForestClassifier algorithm is used for the classification
        - GridSearchCV is used to find the best parameters for the model.
 
    - The data is splitted in two parts:
        - A trainning dataset containing the 80% of the data
        - A test dataset containing the 20% of the data

5. Evaluate the Results
    - The results are evaluated using three metrics:
        - Precision
        - Recall
        - f1 score

6. Deploy
    - The Deploy is a flask web app, where a message can be written and get the message categories, also some statistics about the dataset is showed, this the main app screen:  ![alt text](https://github.com/rafaelmata357/Disaster-Response-Pipeline/blob/main/Fig%203.png 'Main app screen')


## Terms of use:

This app is not a formal tool to classify disaster messages, is a project to show how to use a supervised ML Pipeline to classify messages.

## License:

The code follows this license: https://creativecommons.org/licenses/by/3.0/us/
