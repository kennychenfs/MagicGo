from keras.layers import Input,Dense,Dropout,Flatten,Conv2D,MaxPooling2D,merge,BatchNormalization,Activation,add,Lambda
from keras.models import Model
import numpy as np
import tensorflow as tf
'''
inputs=Input(shape=(13,13,4))
mainmodel=Conv2D(192,kernel_size=(3,3),padding='same')(inputs)
mainmodel=BatchNormalization(axis=3)(mainmodel)
mainmodel=Activation('relu')(mainmodel)
for _ in range(15):
	cnn_1=Conv2D(192,kernel_size=(3,3),padding='same')
	cnn_2=Conv2D(192,kernel_size=(3,3),padding='same')
	bn_1=BatchNormalization(axis=3)
	bn_2=BatchNormalization(axis=3)
	relu=Activation('relu')
	resend=bn_2(cnn_2(relu(bn_1(cnn_1(mainmodel)))))
	mainmodel=relu(add([mainmodel,resend]))
p=Activation('relu')(BatchNormalization(axis=3)(Conv2D(2,(1,1),padding='same')(mainmodel)))
p=Flatten()(p)
p=Dense(13*13+1,activation="softmax",kernel_initializer='random_uniform',bias_initializer='ones',name='policy_head')(p)


v=Activation('relu')(BatchNormalization(axis=3)(Conv2D(1,(1,1),padding='same')(mainmodel)))
v=Flatten()(v)
v=Dense(192,activation='relu',kernel_initializer='random_uniform',bias_initializer='ones')(v)
v=Dense(1,kernel_initializer='random_uniform',bias_initializer='ones')(v)
v=Activation('tanh',name='value_head')(v)

model=Model(inputs=inputs,outputs=[p,v])
model.compile(loss='mse',optimizer='adam')
'''
model = tf.contrib.keras.models.load_model('random.h5')
for i in range(10):
	indata=np.load('randomdata/randomdata'+str(i)+'.npy')
	resultdata=np.load('randomdata/randomdataresult'+str(i)+'.npy')
	chancedata=np.load('randomdata/randomdatachance'+str(i)+'.npy')
	model.fit(x=indata.astype('float32'),y=[chancedata.astype('float32'),resultdata.astype('float32')],epochs=1,verbose=1)
	model.save('random.h5')
