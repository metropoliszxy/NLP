#coding=utf-8
#代码中包含中文，就需要在头部指定编码。

'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
#是一个25000维的向量，每个元素是一个int型的list，表示该样本的word sequence 
print(len(x_test), 'test sequences')
print(x_train[0:1])
print(y_train[0:1])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

'''
调用sequence.pad_sequences对样本进行处理，使得所有样本的word sequence为指定长度（此处为80）
比指定长度短的word sequence填充0
比指定长度长的word sequence直接截断
x_train.shape=(25000, 80)
x_test.shape=(25000, 80)
'''
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print(x_train)
print(y_train)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

'''
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
#batch_size：对总的样本数进行分组，每组包含的样本数量
 #  epochs ：训练次数
print('Test score:', score)
print('Test accuracy:', acc)

'''
