#coding=utf-8
#代码中包含中文，就需要在头部指定编码。

from sklearn import datasets
#from keras.datasets import boston_housing
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,Reshape,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D,Conv1D,MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import os



#1.导入数据集
print("1.导入数据集")
diabetes = datasets.load_diabetes()  
train_data = diabetes.data # 获得其特征向量
train_targets = diabetes.target # 获得样本label

#这是一个糖尿病的数据集，主要包括442行数据，10个属性值，分别是：Age(年龄)、性别(Sex)、Body mass index(体质指数)、
#Average Blood Pressure(平均血压)、S1~S6一年后疾病级数指标。Target为一年后患疾病的定量指标。
print(train_data.shape)
print(train_targets.shape)
print(train_data[:1])
print(train_targets[:1])


#2.将数据集划分为 训练集和测试集
# test_size：　　float-获得多大比重的测试样本 （默认：0.25）　　int - 获得多少个测试样本
# random_state:　　int - 随机种子（种子固定，实验可复现）
#shuffle - 是否在分割之前对数据进行洗牌（默认True）
print("\n2.将数据集划分为 训练集和测试集")
train_data, test_data, train_targets, test_targets = train_test_split(train_data, train_targets, test_size=0.3, random_state=42)
print(train_data.shape)
print(train_targets.shape)
print(train_data[:1])
print(train_targets[:1])

#3.建立模型
print("\n3.建立模型")
def build0():
    model=Sequential(name='diabetes')
    model.add(BatchNormalization(input_shape=(10,)))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build1():
    model=Sequential(name='diabetes')
    model.add(BatchNormalization(input_shape=(10,)))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build2():
    model=Sequential(name='diabetes')
    model.add(BatchNormalization(input_shape=(10,)))
    model.add(Dense(10,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build3():
    model=Sequential(name='diabetes')
    model.add(BatchNormalization(input_shape=(10,)))
    model.add(Dense(10,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build4():
    model=Sequential(name='diabetes')
    model.add(BatchNormalization(input_shape=(10,)))
    model.add(Reshape((10,1)))
    model.add(Conv1D(filters=10,strides=1,padding='same',kernel_size=2,activation='sigmoid'))
    model.add(Conv1D(filters=20, strides=1, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2,strides=1,padding='same'))
    model.add(Conv1D(filters=40, strides=1, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(Conv1D(filters=80, strides=1, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2, strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def build5():
    model = Sequential(name='diabetes')
    model.add(BatchNormalization(input_shape=(10,)))
    model.add(Reshape((10,1)))
    model.add(Conv2D(filters=10, strides=1, padding='same', kernel_size=1, activation='sigmoid'))
    model.add(Conv2D(filters=20, strides=2, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
    model.add(Conv2D(filters=40, strides=1, padding='same', kernel_size=1, activation='sigmoid'))
    model.add(Conv2D(filters=80, strides=2, padding='same', kernel_size=2, activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=2, strides=1, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

#3.训练模型
print("\n3.训练模型")
for i in range(4,5):
    print(("build and train diabetes"+str(i)+" model..."))
    model=eval("build"+str(i)+"()")
    #均方误差（mean-square error, MSE）是反映估计量与被估计量之间差异程度的一种度量。
    #监测的指标为mean absolute error(MAE)平均绝对误差---两个结果之间差的绝对值。
    #model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])adam
    model.compile('adam','mae')
    history=model.fit(train_data,train_targets,batch_size=16,epochs=80,verbose=0,validation_data=(test_data,test_targets))
    #print(len(history.history))
    #print(history.history)
    #print(history.history['loss'])
    #print(history.history['acc'])
    
    # plot train and validation loss
    pyplot.plot(history.history['loss'])
    pyplot.title("model train vs validation loss - build"+str(i)+"()")
    #pyplot.title('model train vs validation loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.legend(['train','validation'],loc='upper right')
    pyplot.show()


'''  
    f=open("result.txt",'a')
    f.write(str(history.history['val_loss'][-1])+"\n")
    f.close()
'''    
    

#查看第一列年龄的数据
#print(diabetes.data[1])



'''
def build_model():
    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    #均方误差（mean-square error, MSE）是反映估计量与被估计量之间差异程度的一种度量。
    #监测的指标为mean absolute error(MAE)平均绝对误差---两个结果之间差的绝对值。
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model


(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
print(mean)
train_data -= mean # 减去均值
std = train_data.std(axis=0) # 特征标准差
print(std)
train_data /= std
#print(train_data)
test_data -= mean #测试集处理：使用训练集的均值和标准差；不用重新计算
test_data /= std
#print(test_data)

print(train_data.shape)
print(test_data.shape)

print("\n")
print(train_data[10:11])
print(train_targets[10:11])

print("\n")
print("train_targets.mean:")
print(train_targets.mean(axis=0))
'''

'''
因为数据各个特征取值范围各不相同，不能直接送到神经网络模型中进行处理。
尽管网络模型能适应数据的多样性，但是相应的学习过程变得非常困难。
一种常见的数据处理方法是特征归一化normalization---减均值除以标准差；数据0中心化，方差为1.
'''

'''
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples] # 划分出验证集部分
    val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]

    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)* num_val_samples:] ],axis=0) # 将训练集拼接到一起
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)* num_val_samples:] ],axis=0)

    model = build_model()
    model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=16,verbose=0)#模型训练silent模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0) # 验证集上评估
    all_scores.append(val_mae)
    
#模型训练

model = build_model()
model.fit(train_data, train_targets,epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)# score 2.5532484335057877   

print("test_mse_score")  
print(test_mse_score)    
print("test_mae_score")
print(test_mae_score)


classes=model.predict(test_data[10:11],batch_size=128)
print(classes)
print(test_targets[10:11])
'''