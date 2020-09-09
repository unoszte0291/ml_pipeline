import sys
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import sqlite3
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier


def load_data(database_filepath):
     '''
    Load data, transform DataFrame, get X, Y and name of feature columns for score results
    
    INPUT:
    engine - create data base
    df - read a table of engine
    
    OUTPUT:
    X - an array with columns messages from df
    Y - a new dataframe that has the following characteristics:
            1. has all comns except for 'id', 'message', 'original', and 'genre'.
            2. has a column 'related' cleaned from values 2.
    category_names - a list of columns names in Y.    
    '''
    
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    engine.table_names()
    df = pd.read_sql_table('df', con=engine)
    df = df.fillna(0)
    # extract values from X and Y
    X =  df['message'].values
    Y = df.loc[:, 'related':'direct_report']
    category_names = Y.columns
    return X, Y,category_names

def tokenize(text):
    '''
    Clean, normalize, tokenize, lemmatize a text
    
    INPUT:
    text - a string, in this case messages
    
    OUTPUT:
    clean_tokens - an array with a text that has the following characteristics:
            1. has no punctuation marks
            2. splited into sequence of words
            3. cleaned from stopwords
            4. lemmatized
            5. all letters are in low case
    '''
    #normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #tokenize
    tokens = word_tokenize(text)
    
    #stop_words
    my_stopwords=stopwords.words('english')
    tokens = [word for word in tokens if word not in my_stopwords]
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Create a pipeline and parameters for a grid search model
    
    OUTPUT:
    cv - model that:
            1. defines an improved pipeline 
            2. sets parameters for estimators
            3. defins a grid search model with the pipeline and parameters
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    # Improved parameters 
    parameters = {
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__learning_rate': [0.1, 0.3]
    }
    # new model with improved parameters
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=3)
    return cv

model = build_model()

def evaluate_model(Y_test, Y_pred):
    '''
    Create a pipeline and parameters for a grid search model
    
    OUTPUT:
    cv - model that:
            1. defines an improved pipeline 
            2. sets parameters for estimators
            3. defins a grid search model with the pipeline and parameters
    '''
      test_pred = model.predict(X_test) 

    for i in range(len(category_names)): 
        print(category_names[i]) 
        print(classification_report(Y_test[category_names[i]], test_pred[:, i]))
     

def save_model(model, model_filepath):

    pickle.dump(model, open('classifier.pkl', 'wb'))



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
