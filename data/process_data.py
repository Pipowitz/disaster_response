import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Takes two filespaths for csvs, loads the data into dataframes and merges them.
    
    inputs:
        -messages_filepath: an url of a csv file
        -categories_filepath: an url of a csv file
        
    returns:
        -df: a pandas dataframe
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    
    return df
    
def clean_data(df):
    '''
    Cleans and converts the dataframe into a useable format.
    
    inputs:
        -df: a pandas dataframe
        
    returns:
        -df: a pandas dataframe
    '''
    
    # create a dataframe of the categories
    categories = df["categories"].str.split(";",expand=True)
    
    # turn the categorie names into column headers
    row = categories[:1].values[0]
    category_colnames = [cat[:-2] for cat in row]
    categories.columns = category_colnames
    
    # extract the binary data from the cells
    for column in categories:
    
        categories[column] = categories[column].apply(lambda x:x[-1]) 
    
        categories[column] = pd.to_numeric(categories[column])
        
        # some of the values are "2" and need to be converted to "1"
        categories[column] = categories[column].apply(lambda x:0 if x==0 else 1)
        
    # drop the categories column, as it's no longer needed
    df.drop("categories", axis=1, inplace=True)
    
    # add the new categories columns
    df = pd.concat([df, categories], axis=1)
    
    #drop duplicates
    df = df[~df.duplicated()]
    
    return df
    
def save_data(df, database_filename):
    '''
    Cleans and converts the dataframe into a useable format.
    
    inputs:
        -df: a pandas dataframe
        -database_filename: the name of the database
        
    '''
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disastertable', engine, index=False, if_exists="replace")  


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