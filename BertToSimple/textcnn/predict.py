# -*-coding:utf-8-*-

import pickle, numpy as np
import time
from keras.layers import *
from keras.models import Model
from keras.initializers import Constant
from keras.preprocessing import sequence
from keras.models import load_model    
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from utils import load_data

def get_textcnn(x_len, v_size, embs):
	x = Input(shape=(x_len,),dtype='int32')
	# embed = Embedding(v_size,300)(x)
	embed = Embedding(v_size,300,embeddings_initializer=Constant(embs),trainable=False)(x)
	cnn1 = Convolution1D(256,3,padding='same',strides=1,activation='relu')(embed)
	cnn1 = MaxPool1D(pool_size=4)(cnn1)
	cnn2 = Convolution1D(256,4,padding='same',strides=1,activation='relu')(embed)
	cnn2 = MaxPool1D(pool_size=4)(cnn2)
	cnn3 = Convolution1D(256,5,padding='same',strides=1,activation='relu')(embed)
	cnn3 = MaxPool1D(pool_size=4)(cnn3)
	cnn = concatenate([cnn1,cnn2,cnn3],axis=-1)
	flat = Flatten()(cnn)
	drop = Dropout(0.2,name='drop')(flat)
	y = Dense(3,activation='softmax')(drop)
	model = Model(inputs=x,outputs=y)
	return model

def get_birnn(x_len, v_size, embs):
	x = Input(shape=(x_len,),dtype='int32')
	# embed = Embedding(v_size,300)(x)
	embed = Embedding(v_size,300,embeddings_initializer=Constant(embs),trainable=False)(x)
	# bi = Bidirectional(GRU(256,activation='tanh',recurrent_dropout=0.2,dropout=0.2,return_sequences=True))(embed)
	bi = Bidirectional(GRU(256,activation='tanh',recurrent_dropout=0.2,dropout=0.2))(embed)
	bi_1 = Bidirectional(GRU(256, activation='tanh', recurrent_dropout=0.2, dropout=0.2))(embed)
	y = Dense(3,activation='softmax')(bi_1)
	model = Model(inputs=x,outputs=y)
	return model


def predict():
	x_len = 50

	# ----- ----- ----- ----- -----
	# from keras.datasets import imdb
	# (x_tr,y_tr),(x_te,y_te) = imdb.load_data(num_words=10000)
	# ----- ----- ----- ----- -----

	name = 'hotel' # clothing, fruit, hotel, pda, shampoo
	(x_tr,y_tr,_),(x_de,y_de,_),(x_te,y_te,_),v_size,embs = load_data(name)
	x_tr = sequence.pad_sequences(x_tr,maxlen=x_len)
	x_de = sequence.pad_sequences(x_de,maxlen=x_len)
	x_te = sequence.pad_sequences(x_te,maxlen=x_len)
	y_tr = to_categorical(y_tr,3)
	y_de = to_categorical(y_de,3)
	y_te = to_categorical(y_te,3)
	#with open('data/cache/t_tr','rb') as fin: y_tr = pickle.load(fin)
	#with open('data/cache/t_de','rb') as fin: y_de = pickle.load(fin)
	# y_tr = to_categorical(y_tr.argmax(axis=1),3)
	# y_de = to_categorical(y_de.argmax(axis=1),3)

	# ----- ----- predict ----- -----
	 # 模型的加载及使用
    
	print("Using loaded model to predict...")
 
	load_model1 = load_model("model_weight(textcnn).h5")
	start= time.time()
	predicted = load_model1.predict(x_te)
	predict_1 = np.argmax(predicted,axis=1)
	print(predict_1.shape)
	end = time.time()
	print('time:',end-start,' acc:',accuracy_score(y_te, predict_1))
    
	load_model1.summary()
	
	# ----- ----- ----- ----- -----

if __name__ == '__main__':
	# run_small()
	predict()
