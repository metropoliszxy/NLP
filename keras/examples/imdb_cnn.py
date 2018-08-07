#coding=utf-8
#代码中包含中文，就需要在头部指定编码。

'''This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs.
90s/epoch on Intel i5 2.4Ghz CPU.
10s/epoch on Tesla K40 GPU.
'''

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
#from keras.datasets import reuters
from keras.utils import plot_model

import numpy as np

# set parameters:
max_features = 5000
#max_features = 1000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
#epochs = 2
epochs = 1


print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(y_train), 'test sequences')

print("x_train\n")
print(x_train)
print("y_train\n")
print(y_train)



print('Pad sequences (samples x time)')
#maxlen设置最大的序列长度，长于该长度的序列将会截短，短于该长度的序列将会填充
#为了实现的简便，keras只能接受长度相同的序列输入。因此如果目前序列长度参差不齐，
#这时需要使用pad_sequences()。该函数是将序列转化为经过填充以后的一个新序列。
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


#引入sequential，这个就是一个空的网络结构，并且这个结构是一个顺序的序列，
#所以叫Sequential，Keras里面还有一些其他的网络结构。
print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
#在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，防止过拟合。
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
#padding 补齐。参数有valid,same,casual
#vaild:：代表有效的卷积，边界数据不处理，same:代表保留边界卷积结果，
#通常导致输出shape与输入的shape相同。causal:将产生因果(膨胀)卷积，
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
#hidden_dims表示输出的维度
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
#Activation：激活层
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
#最后一层用softmax作为激活函数
model.add(Activation('sigmoid'))
# 使用交叉熵作为loss函数
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("The model is built.")

#训练
#   .fit的一些参数
#   batch_size：对总的样本数进行分组，每组包含的样本数量
#   epochs ：训练次数
#   shuffle：是否把数据随机打乱之后再进行训练
#   validation_split：拿出百分之多少用来做交叉验证
#   verbose：屏显模式 0：不输出  1：输出进度  2：输出每次的训练结果
print('Train model...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
validation_data=(x_test, y_test))
print('The model is trained.')

print("该函数将画出模型结构图，并保存成图片：")
plot_model(model, to_file='model.png')



#这个就是评估训练结果。
print('Predict model...')
classes=model.predict(x_test,batch_size=128)
print(classes)


print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(x_train)

'''
#以NumPy专用的二进制类型保存数据，这两个函数会自动处理元素类型和shape等信息，
#使用它们读写数组就方便多了，但是numpy.save输出的文件很难和其它语言编写的程序读入
print('Loading data...')
a=np.arange(5)
np.save('test.npy',a)

a=np.load('test.npy')
print(a)


#引入sequential，这个就是一个空的网络结构，并且这个结构是一个顺序的序列，
#所以叫Sequential，Keras里面还有一些其他的网络结构。
print('Build model...')
model = Sequential()



# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

'''
