import keras
import nltk
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from keras.layers import Dense, Input, Flatten, Dropout, Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from keras.models import model_from_json



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from nltk.tokenize import RegexpTokenizer

import gensim.downloader as api
from gensim.models import KeyedVectors


from help_function import *
from model import *


text = pd.read_table("insight_data.tsv")
text2 = pd.read_table("insight_data_v2.tsv", error_bad_lines=False)
full_text = pd.concat([text,text2])

clean_text = standardize_text(full_text, "text")
clean_text = clean_text.drop(clean_text.index[[2145583, 2496221]])

tokenizer = RegexpTokenizer(r'\w+')

clean_text.text = clean_text.text.astype(str)
clean_text["tokens"] = clean_text["text"].apply(tokenizer.tokenize)
clean_text.head()


all_words = [word for tokens in clean_text["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_text["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))


#word2vec = api.load("word2vec-google-news-300")
word2vec = KeyedVectors.load_word2vec_format('word2vec_model/wiki.en.vec')


# generate the training and testing data
embeddings = get_word2vec_embeddings(word2vec, clean_text)

list_corpus = clean_text["text"].tolist()
list_labels = clean_text["term_selected"].tolist()

X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels, test_size=0.2, random_state=40)


EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 46
VOCAB_SIZE = len(VOCAB)

VALIDATION_SPLIT= 0.2
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(clean_text["text"].tolist())
sequences = tokenizer.texts_to_sequences(clean_text["text"].tolist())

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = (np.asarray(clean_text["term_selected"]))

indices = np.arange(cnn_data.shape[0])
np.random.shuffle(indices)
cnn_data = cnn_data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

embedding_weights = np.zeros((len(word_index)+1, EMBEDDING_DIM))
for word,index in word_index.items():
    embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)

print(embedding_weights.shape)


# define example
data = labels
values = np.array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
print(inverted)



x_train = cnn_data[:-num_validation_samples]
y_train = onehot_encoded[:-num_validation_samples]
x_val = cnn_data[-num_validation_samples:]
y_val = onehot_encoded[-num_validation_samples:]

model = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index)+1, EMBEDDING_DIM, 
                len(list(clean_text["term_selected"].unique())), True)

json_string = model.to_json()
open('model_architecture.json', 'w').write(json_string)


checkpointer = ModelCheckpoint(filepath='results/model_trainable.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=256, verbose = 2, callbacks=[checkpointer])


# Test on validation set for the top3 and top5 results
model = model_from_json(open('model_architecture.json').read())
model.load_weights('results/model_trainable.h5')

y_pred = model.predict(x_val)

truth = np.where(y_val == 1.)[1]
order_pred = np.argsort(-y_pred,axis=1)

# Top1
pred = np.argmax(y_pred, axis = 1)
count = len(np.where(pred == truth)[0])
print("Prediction is the Top1 results:" + str(count/num_validation_samples))

# Top3
top = [l[:3] for l in order_pred]
count = 0 
for i in range(len(order_pred)):
    if truth[i] in top[i]:
        count += 1
        
print("Prediction is the Top3 results:" + str(count/num_validation_samples))

# Top5
top = [l[:5] for l in order_pred]
count = 0 
for i in range(len(order_pred)):
    if truth[i] in top[i]:
        count += 1
        
print("Prediction is the Top5 results:" + str(count/num_validation_samples))

