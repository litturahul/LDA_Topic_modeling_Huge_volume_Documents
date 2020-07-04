import os
import pandas as pd
import tika
tika.TikaClientOnly = True
from tika import parser
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
nltk.download('wordnet')

#Extraction of data from pdf
folder="C:/Users/Abhinay/Documents/Moodle/RGP/paperz/"
filename = []
content = []

for file in os.listdir(folder):
    print(file)
    file_data = parser.from_file(folder+file)
    try:
        text = file_data['content'].replace("\n", " ").replace(file_data['metadata']['title'], " ")
    except:
        text = file_data['content'].replace("\n", " ")
    file_name = file_data['metadata']['resourceName']
    filename.append(file_name)
    content.append(text)

df=pd.DataFrame({'filename':filename,'content':content})

#Filtering the text
df['content'] = df['content'].apply(lambda x: x.split("Reference")[0])
df['content'] = df['content'].apply(lambda x: x.strip().lower())
df['content'] = df['content'].apply(lambda x: re.sub(r"(\w+@\w+.\w+)|(http)[^\s]+","",x))
df['content'] = df['content'].str.replace(r"(\.|,|\?|!|@|#|\$|%|\^|&|\*|\(|\)|_|-|\+|=|;|:|~|`|\d+|\[|\]|{|}|\xA9|\\|\/)","")
df['content'] = df['content'].apply(lambda x: re.sub(r"\"|\'","",x))

#Performing lemmatization, stemming & Tokenization
def lemmatize_stemming(text):
    return SnowballStemmer(language="english").stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#Applying the cleaning process on data
processed_corpus=df['content'].map(preprocess)

#Creating dictionary
dictionary = gensim.corpora.Dictionary(processed_corpus)

#Vectorzing the dictionary
vector=[dictionary.doc2bow(doc) for doc in processed_corpus]

# tfidf model building
from gensim import models
tfidf_model = models.TfidfModel(vector)

#Creating weighted vector of the whole data
corpus_tfidf = tfidf_model[vector]

for doc in corpus_tfidf:
    print(doc)

#Lda model
model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)

for idx, topic in model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))