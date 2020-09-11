import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = messages.merge(categories_df, how='left',on=['id'])
    return df

def clean_data(df):
    #split the categories and create a dataframe of individual categories columns
    categories_df = df['categories'].str.split(pat=';', expand=True)
    #select the first row of the categories dataframe
    row = categories_df.iloc[0]
    #use the row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x[:-2])
    # rename the columns of `categories`
    categories_df.columns = category_colnames 
    for column in categories_df:
       categories_df[column] = categories_df[column].apply(lambda x:x[-1:])        
       categories_df[column] = categories_df[column].astype(int)
       categories_df.drop(categories_df.index[categories_df.related == 2], inplace=True) 
    # drop the original categories column from `df`
       df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
       df = pd.concat([df, categories_df], axis=1)
    # drop duplicates
       df.drop_duplicates(inplace=True)
       df = df.fillna(0)
   
       return df

def save_data(df, database_filename):
    database_filename = './DisasterResponse.db'
    engine = create_engine('sqlite:///' + database_filename)
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