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
    discarded_messages = 0
    messages = channel['text_messages']
    len_messages = len(messages)
    messages = [messages[key]['message'] for key in messages if len(messages[key]['message']) > 25]
    discarded_messages += len_messages - len(messages)
    ok_messages = []
        
    for message in messages:
        if len(ok_messages) >= 100:
            break
        english = False
        if english == True:
            try:
                if detect(message)=='en': ok_messages.append(message)
                else: discarded_messages +=1
            except:
                pass
        else:
            ok_messages.append(message)
        
    single_corpus = ' '.join(ok_messages)

    return (single_corpus, _id, discarded_messages, ok_messages)

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

    done_prev = open_pickle(f'processed_channels')
    done_channels = done_prev.keys()
    df = df[~df['ch_id'].isin(done_channels)]
    done_numbers = max(done_prev.values())
    iteration = done_numbers+1

    english = False
    if english == True:
        df_ = df[df['language']=='en']
        channels = list(df_['ch_id'])
    channels = list(df['ch_id'])
    portions = split_list(channels, portion_size)
    # print(len(channels))

    done = done_prev

    # print('Starting preprocessing')
    for i, portion in tqdm(enumerate(portions), total=len(portions)):
        print(len(portion))
        # print('Getting corpus')
        # print('Getting channels')
        channels = db_utilities.get_channels_by_ids(portion, db_name='Telegram_test')
        # print('Getting messages')
        corpus = []
        all_messages = []
        id_list = []
        discarded_messages = 0  
        with Pool(n_pool) as pool:
            for single_corpus, _id, s_discarded_messages, ok_messages in pool.map(get_corpus, channels):
                corpus.append(single_corpus)
                id_list.append(_id)
                all_messages.append(ok_messages)
                discarded_messages += s_discarded_messages

        save_as_pickle(id_list, f'ids_list_topic_modeling/n_gram_ids_list_topic_modeling_{i+iteration}')
        save_as_pickle(discarded_messages, f'discarded_messages_topic_modeling/n_gram_discarded_messages_topic_modeling_{i+iteration}')
        save_as_pickle(corpus, f'corpus/n_gram_corpus_{i+iteration}')
        save_as_pickle(all_messages, f'messages_per_channel/messages_{i+iteration}')
        # print(len(all_messages))
        # print('Calculating embeddings')
        embeddings = []
        for channel_messages in tqdm(all_messages):
            channel_embedding = bert_preprocess(channel_messages)
            embeddings.append(channel_embedding)
        
        save_as_pickle(embeddings, f'texts_bert/texts_topic_modeling_{i+iteration}')

        for channel in portion:
            done[channel] = i

        save_as_pickle(done, f'processed_channels')

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained("bert-base-multilingual-cased")
    preprocessing_bert(1000, 1)
