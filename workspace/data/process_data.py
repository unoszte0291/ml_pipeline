import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    messages_filepath = 'disaster_messages.csv'
    categories_filepath = 'disaster_categories.csv'
    messages = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages.merge(categories_df, how='left',on=['id'])
    return df

categories_filepath = 'disaster_categories.csv'
categories_df = pd.read_csv(categories_filepath)
categories_df = categories_df.categories.str.split(';', expand=True)

def clean_data(df):
 row = categories_df.iloc[0,:]
 category_colnames = row.apply(lambda x:x[:-2])
 categories_df.columns = category_colnames
 for column in categories_df:
    # set each value to be the last character of the string
  categories_df[column] = categories_df[column].str[-1]
    
    # convert column from string to numeric
 categories_df[column] = categories_df[column].astype(int) 
 categories_df.drop(categories_df.index[categories_df.related == 2], inplace=True)
 df = df.drop('categories',axis=1)
 df = pd.concat([df,categories_df],axis=1)
 df = df.drop_duplicates()
 return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('df', engine, index=False, if_exists='replace')  


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