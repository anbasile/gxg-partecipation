"""
A library that loads the datasets that we need for our experiments.
"""

import logging
import pandas as pd
import os.path

logging.basicConfig(level=logging.DEBUG)

def load_yelp():
    """
    Load the Yelp Dataset from json
    """
    logging.info("Loading the Yelp dataset...")
    df = pd.read_json('../data/review_lang_annotated.json', chunksize=1, lines=True)

    for chunk in df:
        for index, row in chunk.iterrows():
            yield row.text, row.business_id
    #raise NotImplementedError

def load_amazon():
    """
    Load the Amazon Reviews Dataset
    """
    raise NotImplementedError

def load_labov(dataset='labov05'):
    try:
        logging.info('Loading the dataset %s', dataset)
        df = pd.read_csv('../data/labov05.csv',
                         sep=',',
                         encoding='utf-8',
                         index_col=0,
                         names=['text', 'label'],
                         skiprows=1,
                         dtype={'text':str, 'label':float}).dropna().reset_index(drop=True)
        df.label = df.label.astype(str)
        logging.info('Ok, I have loaded %s. It has shape %s', dataset, str(df.shape))
        logging.info('These are the labels: %s', set(df.label))
    except:
        df = None
        logging.info('Error loading the dataset!')
    return df
