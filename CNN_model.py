# Test different window length
from __future__ import print_function
import numpy as np
import time
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.applications import *
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.metrics import precision_score, recall_score, classification_report
from utils.plot import Plot
from matplotlib import pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from contextlib import redirect_stdout
from keras.utils.vis_utils import plot_model

from sklearn.metrics import precision_recall_fscore_support as score

Length = 400  # length of window

def load_data():

	labels = np.loadtxt('label.txt')
	encoded_seq = np.loadtxt('encoded_seq.txt')
	
	x_train,x_test,y_train,y_test = train_test_split(encoded_seq,labels,test_size=0.1)

	return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)



def resnet_model():
		

	resnet50_imagenet_model = ResNet50(include_top=False, weights='imagenet', batch_input_shape=(None, Length, 4))

	#Flatten output layer of Resnet
	flattened = tf.keras.layers.Flatten()(resnet50_imagenet_model.output)


	#Fully connected layer 1
	fc1 = tf.keras.layers.Dense(128, activation='relu', name="AddedDense1")(flattened)

	#Fully connected layer, output layer
	fc2 = tf.keras.layers.Dense(12, activation='softmax', name="AddedDense2")(fc1)

	model = tf.keras.models.Model(inputs=resnet50_imagenet_model.input, outputs=fc2)

	adam = Adam(lr=1e-4)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

	return model




def load_model():

    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    
    tensor_in = Input((60, 200, 3))
    out = tensor_in
    out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
    out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = Flatten()(out)
    out = Dropout(0.5)(out)
    out = [Dense(37, name='digit1', activation='softmax')(out),\
        Dense(37, name='digit2', activation='softmax')(out),\
        Dense(37, name='digit3', activation='softmax')(out),\
        Dense(37, name='digit4', activation='softmax')(out),\
        Dense(37, name='digit5', activation='softmax')(out),\
        Dense(37, name='digit6', activation='softmax')(out)]
    
    model = Model(inputs=tensor_in, outputs=out)
    
    # Define the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    if 'Windows' in platform.platform():
        model.load_weights('{}\\cnn_weight\\verificatioin_code.h5'.format(PATH)) 
    else:
        model.load_weights('{}/cnn_weight/verificatioin_code.h5'.format(PATH)) 
    
    return model 



def deep_model():
	# build the model
	model = Sequential()
	model.add(keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, Length, 4), activation='relu'))

	model.add(keras.layers.Conv2D(32, kernel_size =(9, 9), strides =(1, 1), padding='same', data_format="channels_last", 
					activation ='relu'))
	model.add(MaxPooling2D(pool_size =(2, 2), strides =(2, 2)))
	model.add(keras.layers.Conv2D(64, (9, 9), activation ='relu'))
	model.add(MaxPooling2D(pool_size =(2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation ='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(3, activation ='softmax'))

	# training the model
	model.compile(loss = keras.losses.categorical_crossentropy,
				optimizer = keras.optimizers.SGD(lr = 0.01),
				metrics =['accuracy'])


	return model




def deep_cnn_classifier():

	model = Sequential()

	model.add(keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, Length, 4), activation='relu'))

	model.add(Flatten())
	model.add(Dense(100,activation='relu'))
	
	model.add(Dropout(0.3))
	model.add(Dense(3,activation='softmax'))
	
	adam = Adam(lr=1e-4)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	return model




def cnn_classifier():

	model = Sequential()
		

	model.add(keras.layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, Length, 4), activation='relu'))
	model.add(keras.layers.Conv1D(filters=50, kernel_size=7, strides=1, padding='same', batch_input_shape=(None, Length, 4), activation='relu'))
	model.add(keras.layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='same', batch_input_shape=(None, Length, 4), activation='relu'))


	model.add(Flatten())
	model.add(Dense(100,activation='relu'))
	
	model.add(Dropout(0.3))
	model.add(Dense(3,activation='softmax'))
	
	adam = Adam(lr=1e-4)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
	return model


def training_process(x_train,y_train,x_test,y_test):

	# x_train = x_train.reshape(-1, Length, Length, 4)


	x_train = x_train.reshape(-1, Length, 4)
 
	y_train = np_utils.to_categorical(y_train, num_classes=3)
	x_test = x_test.reshape(-1, Length, 4)
	y_test = np_utils.to_categorical(y_test, num_classes=3)
	print(x_train.shape, y_train.shape)
	
	epoch = 3
	print("======================")
	print('Convolution Neural Network')
	x_plot = list(range(1,epoch+1))
	start_time = time.time()
	model = deep_model()
	history = model.fit(x_train, y_train, epochs=epoch, batch_size=50)
	model.save('CNN.h5')
	
	loss,accuracy = model.evaluate(x_test,y_test)
	print(model.summary())
	with open('modelsummary.txt', 'w') as f:
		with redirect_stdout(f):
			model.summary()
	print('testing accuracy: {}'.format(accuracy))
	print('testing loss: {}'.format(loss))
	print('training took %fs'%(time.time()-start_time))




	prob = model.predict(x_test)
	predict = model.predict(x_test)
	predict = np_utils.to_categorical(predict, num_classes=3)
	y_true = y_test



	# pred = model.predict(x_test, batch_size=32, verbose=1)
	predicted = np.argmax(prob, axis=1)
	report = classification_report(np.argmax(y_true, axis=1), predicted, output_dict=True )
	print(report)

	macro_precision =  report['macro avg']['precision'] 
	macro_recall = report['macro avg']['recall']    
	macro_f1 = report['macro avg']['f1-score']
	class_accuracy = report['accuracy']

	print("precision: ", macro_precision, "recall: ", macro_recall, "f1: ", macro_f1, "accuracy: ", class_accuracy)



 
	true_0 = y_true[:,0]
	prob_0 = prob[:,0]
	predict_0 = predict[:,0]

	predicted = np.argmax(prob_0)
	# report = classification_report(np.argmax(true_0), predict_0)
	# print("/n/n This is report for acceptor site :", report)
	# auc = metrics.roc_auc_score(true_0,prob_0)
	# precision = metrics.precision_score(true_0,predict_0)
	# recall = metrics.recall_score(true_0,predict_0)
	# f1 = metrics.f1_score(true_0,predict_0)
	# print('acceptor AUC of :%f'%auc)
	# print('acceptor precision of :%f'%precision)
	# print('acceptor recall of :%f'%recall)
	# print('acceptor f1 of :%f'%f1)

	
	# true_1 = y_true[:,1]
	# prob_1 = prob[:,1]
	# predict_1 = predict[:,1]

	# predicted = np.argmax(prob_1, axis=1)
	# report = classification_report(np.argmax(true_1, axis=1), predict_1, **output_dict=True**)




	# print("/n/n This is report for donor site :", report)
	# auc = metrics.roc_auc_score(true_1,prob_1)
	# precision = metrics.precision_score(true_1,predict_1)
	# recall = metrics.recall_score(true_1,predict_1)
	# f1 = metrics.f1_score(true_1,predict_1)
	# print('donor AUC of :%f'%auc)
	# print('donor precision of :%f'%precision)
	# print('donor recall of :%f'%recall)
	# print('donor f1 of :%f'%f1)






	plt.plot(history.history['accuracy'])
	# plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy for CNN model')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	return




def main():
	x_train,y_train,x_test,y_test = load_data()
	
	training_process(x_train,y_train,x_test,y_test)



if __name__ == '__main__':
	main()
