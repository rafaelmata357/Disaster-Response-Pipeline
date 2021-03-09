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


## Instructions to run the app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## How To Interact With Your Project 

- The project in this repository follows the CRISPM methodology:

1. Business Understanding

    - Starting in august 2008 Airbnb  an American company  introduced an app for online lodging disrupting the market,  primarily for home stays  this app offers  the house owners the opportunity to rent available house rooms and give the guests different tourism experiences, this model has increased year over year and Boston is not the exception

2. Data Understanding
    - The dataset contains three tables:
        - Listings : with the listings offering, contains 3585 listings for 25 neighborehoods with more than 90 features 
        - Calendar : with the listings availability, contains more than 643000 entries
        - Reviews  : with the listings reviews, contains more than 68000 reviews
    - Different descriptive statistics are used for EDA and data visualization:
        - Histograms
        - WordClouds
        - Choroplet maps
        - Box plots
        - Heat Maps

3. Data preparation
    - A data cleaning process is followed including:
        - Removed columns with unique values
        - Inputing: the NaN values are filled with the most common value for categorical features, and the average for numerical features
        - Remove the outliers
        - Codify the categorical features
        - Use NLTK library to extract the main feature from the transit and neighborhood description columns

4. Data Modeling
    - To model a linear regression two methods are used:
        - Linear regression
        - Random Forest regression
    - The dataset is normalized using Min-Max scaler
    - The data is splitted in two parts:
        - A trainning dataset containing the 80% of the data
        - A test dataset containing the 20% of the data

5. Evaluate the Results
    - The results are evaluated using three metrics:
        - MAS Error
        - MSE Error
        - R2 Score

6. Deploy
    - The Deploy is an article posted in [medium](https://medium.com/) and can be found [here](https://rafaelmata357.medium.com/a-guide-tour-to-the-boston-airbnbs-market-d7689ddf9d6c)


## Terms of use:

This notebook is not a formal study, is an initial analyzis and is made only to show how to use the CRISP Methodology to understand a dataset and answer questions related to the Boston  Airbnb market, . 

## License:

The code follows this license: https://creativecommons.org/licenses/by/3.0/us/
