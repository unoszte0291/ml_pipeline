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
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    
    #tokenize
    words = word_tokenize (text)
    
    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return words_lemmed


def build_model():
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
     Y_pred = pipeline.predict(X_test)

     labels = np.unique(Y_pred)
     confusion_mat = confusion_matrix(Y_test.values.argmax(axis=1), Y_pred.argmax(axis=1),labels=labels)
     accuracy = (Y_pred == Y_test).mean()

     print("Labels:", labels)
     print("Confusion Matrix:\n", confusion_mat)
     print("Accuracy:", accuracy)

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