
# coding: utf-8

# In[1]:

import sys
import tensorflow as tf
from termcolor import colored

print(colored('Python Version: %s' % sys.version.split()[0], 'blue'))
print(colored('Tensorflow Ver: %s' % tf.__version__, 'magenta'))


# In[2]:

n_epoch = int(input('Enter no. of epochs for RNN training:'))


# In[3]:

print(colored('No. of epochs: %d' % n_epoch, 'red'))


# In[4]:

import pandas as pd
pd.set_option('display.max_colwidth', 1000)


# In[5]:

original_ign = pd.read_csv('ign.csv')
original_ign.head(10)


# In[6]:

print('original_ign.shape:', original_ign.shape)


# In[7]:

original_ign.score_phrase.value_counts()


# In[8]:

bad_phrases = ['Bad', 'Awful','Painful', 'Unbearable', 'Disaster']
original_ign['sentiment'] = original_ign.score_phrase.isin(bad_phrases).map({True: 'Negative', False: 'Positive'})


# In[9]:

original_ign = original_ign[original_ign['score_phrase'] != 'Disaster']


# In[10]:

original_ign.head()


# In[11]:

original_ign.sentiment.value_counts(normalize=True)


# In[12]:

original_ign.isnull().sum()


# In[13]:

original_ign.fillna(value=' ', inplace=True)


# In[14]:

ign = original_ign[['sentiment', 'score_phrase', 'title', 'platform', 'genre', 'editors_choice']].copy()
ign.head(10)


# In[15]:

ign['is_editors_choice'] = ign['editors_choice'].map({'Y': 'editors_choice', 'N': ''})
ign.head()


# In[16]:

ign['text'] = ign['title'].str.cat(ign['platform'], sep=' ').str.cat(ign['genre'], sep=' ').str.cat(ign['is_editors_choice'], sep=' ')


# In[17]:

print('Shape of \"ign\" DataFrame:', ign.shape)


# In[18]:

ign.head(10)


# In[19]:

x = ign.text
y = ign.score_phrase


# In[20]:

x.head(10)


# In[21]:

y.head(10)


# In[22]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


# In[23]:

vect = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b')
dummy = DummyClassifier(strategy='most_frequent', random_state=0)
dummy_pipeline = make_pipeline(vect, dummy)


# In[24]:

dummy_pipeline.named_steps


# In[25]:

cv = cross_val_score(dummy_pipeline, x, y, scoring='accuracy', cv=10, n_jobs=-1)
print(colored('\nDummy Classifier\'s Accuracy: %0.5f\n' % cv.mean(), 'yellow'))


# In[26]:

from sklearn.naive_bayes import MultinomialNB


# In[27]:

vect = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w{2,}\b', min_df=1, max_df=0.1, ngram_range=(1,2))
mnb = MultinomialNB(alpha=2)
mnb_pipeline = make_pipeline(vect, mnb)


# In[28]:

mnb_pipeline.named_steps


# In[29]:

cv = cross_val_score(mnb_pipeline, x, y, scoring='accuracy', cv=10, n_jobs=-1)
print(colored('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % cv.mean(), 'green'))


# In[30]:

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


# In[31]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


# In[32]:

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')


# In[33]:

vect.fit(x_train)
vocab = vect.vocabulary_


# In[34]:

def convert_x_to_x_word_ids(x):
    return x.apply( lambda x: [vocab[w] for w in [w.lower().strip() for w in x.split()] if w in vocab])


# In[35]:

x_train_word_ids = convert_x_to_x_word_ids(x_train)
x_test_word_ids = convert_x_to_x_word_ids(x_test)


# In[36]:

x_train.head()


# In[37]:

x_train_word_ids.head()


# In[38]:

print('x_train_word_ids.shape:', x_train_word_ids.shape)
print('x_test_word_ids.shape:', x_test_word_ids.shape)


# In[39]:

x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=20, value=0)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=20, value=0)


# In[40]:

print('x_train_padded_seqs.shape:', x_train_padded_seqs.shape)
print('x_test_padded_seqs.shape:', x_test_padded_seqs.shape)


# In[41]:

pd.DataFrame(x_train_padded_seqs).head()


# In[42]:

pd.DataFrame(x_test_padded_seqs).head()


# In[43]:

unique_y_labels = list(y_train.value_counts().index)
unique_y_labels


# In[44]:

len(unique_y_labels)


# In[45]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(unique_y_labels)


# In[46]:

print(' ')
for label_id, label_name in zip(le.transform(unique_y_labels), unique_y_labels):
    print('%d: %s' %(label_id, label_name))
print(' ')


# In[47]:

y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), nb_classes=len(unique_y_labels))
y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), nb_classes=len(unique_y_labels))


# In[48]:

y_train[0:3]


# In[49]:

print('y_traing.shape:',y_train.shape)
print('y_test.shape:',y_test.shape)


# In[50]:

size_of_each_vector = x_train_padded_seqs.shape[1]
vocab_size = len(vocab)
no_of_unique_y_labels = len(unique_y_labels)


# In[51]:

print('size_of_each_vector:',size_of_each_vector)
print('vocab_size:',vocab_size)
print('no_of_unique_y_labels:',no_of_unique_y_labels)


# In[52]:

net = tflearn.input_data([None, size_of_each_vector])
net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.6)
net = tflearn.fully_connected(net, no_of_unique_y_labels, activation='softmax')
net = tflearn.regression(net,
                        optimizer='adam',
                        learning_rate=1e-4,
                        loss='categorical_crossentropy')


# In[53]:

model = tflearn.DNN(net, tensorboard_verbose=0)


# In[60]:

model.fit(x_train_padded_seqs, y_train,
          validation_set=(x_test_padded_seqs, y_test),
          n_epoch=n_epoch,
          show_metric=True,
          batch_size=100)


# In[55]:

model.save('SavedModels/model.tfl')
#model.load('SavedModels/model.tfl')
print(colored('Model Saved!', 'red'))


# In[56]:

import numpy as np
from sklearn import metrics


# In[58]:

pred_classes = [np.argmax(i) for i in model.predict(x_test_padded_seqs)]
true_classes = [np.argmax(i) for i in y_test]

print(colored('\nRNN Classifier\'s Accuracy: %0.5f\n' %metrics.accuracy_score(true_classes, pred_classes), 'cyan'))


# In[61]:

ids_of_titles = range(0,21)
for i in ids_of_titles:
    pred_class = np.argmax(model.predict([x_test_padded_seqs[i]]))
    true_class = np.argmax(y_test[i])
    
    print(x_test.values[i])
    print('pred_class:', le.inverse_transform(pred_class))
    print('true_class:', le.inverse_transform(true_class))
    print(' ')


# In[ ]:



