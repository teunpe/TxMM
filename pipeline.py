import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
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
    embeddings = []
    for batch in batches:
        # tokenize 
        messages = tokenizer(batch, return_tensors='pt', max_length=512, truncation=True, padding=True)
        # compute embedding
        embedding = model(**messages)['pooler_output'].detach()
        for e in embedding:
            embeddings.append(e.numpy())
    # return embeddings
    mean = np.mean(embeddings, axis=0)
    return mean

def preprocessing_bert(portion_size=1000, n_pool=1):
    # load list of all channels
    df = pd.read_csv('TGDataset/labeled_data/channel_to_language_mapping.csv', sep='\t')

    # load list of channels in current database
    channels_in_db = db_utilities.get_channel_ids(db_name='Telegram_test')

    # load previously preprocessed channels
    done_prev = open_pickle(f'test/done_ids')

    # remove done channels and channels not in database
    df = df[~df['ch_id'].isin(done_prev)]
    df = df[df['ch_id'].isin(channels_in_db)]

    channels = list(df['ch_id'])
    portions = split_list(channels, portion_size) # prepare portions
    
    done = {}
    for i, portion in tqdm(enumerate(portions), total=len(portions)):
        # get channel content
        channels = db_utilities.get_channels_by_ids(portion, db_name='Telegram_test')
        if len(channels) == 0:
            continue
        messages = []
        embeddings = []
        ids = []

        for channel in tqdm(channels):
            ok_messages, _id = get_corpus(channel) # get messages
            embedding = bert_preprocess(ok_messages) # compute embeddings

            messages.append(ok_messages)
            embeddings.append(embedding)
            ids.append(_id)
            done[_id] = 1
        portiondict = {'id': ids, 'embeddings': embeddings, 'messages': messages}
        save_as_pickle(portiondict, f'test/run5_{i}') # save embeddings and messages to a file

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    preprocessing_bert(100, 1)
