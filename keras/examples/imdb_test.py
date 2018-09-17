#coding=utf-8
#代码中包含中文，就需要在头部指定编码。

import keras
from keras.models import Sequential 
from keras.layers import Dense 
import numpy as np

batch_size=32

x=np.array([[0,1,0],[0,0,1],[1,3,2],[3,2,1]])
y=np.array([0,0,1,1]).T

print(x)
print(y)
print(x.shape[1])


simple_model=Sequential()
simple_model.add(Dense(5,input_shape=(x.shape[1],),activation='relu',name='layer1'))
simple_model.add(Dense(4,activation='relu',name='layer2'))
simple_model.add(Dense(1,activation='sigmoid',name='layer3'))

print('Compile...')
simple_model.compile(optimizer='sgd',loss='mean_squared_error')

'''
print('Train...')
simple_model.fit(x,y,epochs=2000,verbose=1,validation_data=(x,y))

simple_model.save("simple_model.mdl")
'''

simple_model.load_weights("simple_model.mdl")

print("result2 = simple_model.predict(x)")
print(x)
result1 = simple_model.predict(x)
print(result1)


print("evaluate:")
#score, acc = simple_model.evaluate(10)
#batch_size：对总的样本数进行分组，每组包含的样本数量
 #  epochs ：训练次数
#print('Test score:', score)
#print('Test accuracy:', acc)


'''
print("result2 = simple_model.predict(x[0:3])")
print(x[0:3])
result2 = simple_model.predict(x[0:3])
print(result2)
'''
