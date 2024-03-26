from langdetect import detect
import re
from tqdm.auto import tqdm
import pickle
import pandas as pd
import unicodedata
import tmtoolkit
import numpy as np
from gensim.parsing.preprocessing import strip_punctuation
from TGDataset import db_utilities
import spacy


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from multiprocessing import Pool
from transformers import BertTokenizer, BertModel
import torch
import os

def open_pickle(filename):
    with open ('/media/teun/Hard Driver/TxMM/preprocessed_docs/'+filename, 'rb') as fp:
        saved_file = pickle.load(fp)

    return saved_file

def split_list(data, partition_size):
    return [data[i: i+partition_size] if i+partition_size< len(data) else data[i: len(data)] for i in range(0, len(data), partition_size)]

def save_as_pickle(text_list, outfile_name):
    with open('/media/teun/Hard Driver/TxMM/preprocessed_docs/'+outfile_name, 'wb') as fp:
        pickle.dump(text_list, fp)

def get_corpus(channel):
    _id = channel['_id']
    messages = channel['text_messages']
    messages = [messages[key]['message'] for key in messages if len(messages[key]['message']) > 25]
    ok_messages = []
        
    for message in messages:
        if len(ok_messages) >= 100:
            break
        else:
            ok_messages.append(message)

    return (ok_messages, _id)

def bert_preprocess(channel, batch_size=2):
    batches = split_list(channel, batch_size)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # model = BertModel.from_pretrained("bert-base-multilingual-cased")
    # print('calc embeddings')
    embeddings = []
    for batch in batches:
        # print(batch)
        messages = tokenizer(batch, return_tensors='pt', max_length=512, truncation=True, padding=True)
        embedding = model(**messages)['pooler_output'].detach()
        for e in embedding:
            embeddings.append(e.numpy())
    # return embeddings
    mean = np.mean(embeddings, axis=0)
    return mean

def preprocessing_bert(portion_size=1000, n_pool=1):
    # print('Loading Bert')
    # print('Getting channels')
    df = pd.read_csv('TGDataset/labeled_data/channel_to_language_mapping.csv', sep='\t')

    # done_prev = open_pickle(f'processed_channels')
    # done_channels = done_prev.keys()
    # df = df[~df['ch_id'].isin(done_channels)]
    # done_numbers = max(done_prev.values())
    # iteration = done_numbers+1

    # english = False
    # if english == True:
    #     df_ = df[df['language']=='en']
    #     channels = list(df_['ch_id'])
    channels = list(df['ch_id'])
    portions = split_list(channels, portion_size)
    # # print(len(channels))
    done = {}
    # done = done_prev

    # print('Starting preprocessing')
    for i, portion in tqdm(enumerate(portions), total=len(portions)):
        channels = db_utilities.get_channels_by_ids(portion, db_name='Telegram_test')
        messages = []
        embeddings = []
        ids = []
        discarded_messages = 0  
        # print(channels)

        for channel in tqdm(channels):
            ok_messages, _id = get_corpus(channel)
            embedding = bert_preprocess(ok_messages)

            messages.append(ok_messages)
            embeddings.append(embedding)
            ids.append(_id)
            done[_id] = 1
        portiondict = {'id': ids, 'embeddings': embeddings, 'messages': messages}
        save_as_pickle(portiondict, f'test/{i}')

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    preprocessing_bert(1000, 1)
