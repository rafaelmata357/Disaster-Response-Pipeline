#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER:    Rafael Mata M.
# DATE CREATED:  9 Marh 2021                                
# REVISED DATE:  23 March 2021
# PURPOSE: A flask web app to run the disaster messages classifier
#
#  Usage: python run.py
#
#

# Imports python modules


import json
import plotly
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from nltk.corpus import stopwords

app = Flask(__name__)

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

# load data
database_filepath = '../data/DisasterMessages.db'
engine = create_engine('sqlite:///{}'.format(database_filepath))
df = pd.read_sql('SELECT * FROM MESSAGES', engine)

# load model
model = joblib.load("../models/model_pkl.bz2")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    categories = df.iloc[:,4:]

    #Calculate the mean by category counts
    category_mean = categories.mean().sort_values(ascending=False)
    category_names = category_mean.index

    #Calculate the average
    messages = df[['message','genre']].copy()
    messages['men_len'] = messages['message'].apply(lambda x: len(x))

    messages_mean = messages.groupby('genre')['men_len'].mean()
    genre_name = messages_mean.index

    
    # create visuals
    # 
    graphs = [
        # Graph #1 Distribution by Genre
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Graph #2 Distribution by Categories
         {
            'data': [
                Bar(
                    x=category_names,
                    y=category_mean
                )
            ],

            'layout': {
                'title': 'Distribution by Categories',
                'yaxis': {
                    'title': "%"
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle': 35
                }
            }
        },
        # Graph #3 Average message length by genre
        {
            'data': [
                Bar(
                    x=genre_name,
                    y=messages_mean
                )
            ],

            'layout': {
                'title': 'Average message length by Genre',
                'yaxis': {
                    'title': "Chars"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()