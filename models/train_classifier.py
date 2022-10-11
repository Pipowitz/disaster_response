import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import fbeta_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    Loads the data from a local database
    
    inputs:
        -database_filepath: A path to the local database
        
    returns:
        -X: The feature data
        -Y: The datas target variable 
        -category_names: The names of the feature datas categories
    '''
    
    # connect to the db and extract the data
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Disastertable', engine)
    
    # split the data into features and targets
    X = df["message"]
    Y = df.iloc[:,4:]
    
    # extract the category names
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    '''
    Tokenizes and lemmatizes the text
    
    inputs:
        -text: a string of words
        
    returns:
        -clean_tokens: a list of tokens
    '''
    
    tokens=word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    
    # lemmantize the tokens
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds a ML pipeline
    
    returns:
        -pipeline: a ML pipeline
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Iterates through the columns and prints the f1 score, precision and recall for each one.
    
    inputs:
        -model: A ML model meant to be evaluated
        -X_test: The models feature data
        -Y_test: The test datas target variable
        -category_names: The names of the categories meant to be evaluated
    '''
    
    y_pred=model.predict(X_test)
    
    for i in enumerate(category_names):
        print(i[1])
        print(classification_report(Y_test[i[1]].values, y_pred[:, i[0]]))


def save_model(model, model_filepath):
    '''
    saves the model in a pickle file.
    
    inputs:
        -model: a ML model
        -model_filepath: the local path where the model is meant to be saved
    '''
    
    with open(model_filepath,'wb') as f:
        pickle.dump(model, f)


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