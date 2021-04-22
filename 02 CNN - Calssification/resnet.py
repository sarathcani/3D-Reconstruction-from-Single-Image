### RESNET - 50 Keras Implementation

import numpy as np 
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input , Add , Dense , Activation , BatchNormalization , Flatten , Conv2D , AveragePooling2D , ZeroPadding2D ,MaxPooling2D
from keras.models import Model , load_model
from keras.preprocessing import image
from keras.initializers import glorot_uniform
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

class Resnet50 :
	def __init__(self , input_shape = (137,137,4) , classes = 4 ) :
		self.model = self.build_model(input_shape , classes)

	def build_model(self , input_shape , classes) :

		X_in = Input(input_shape)
		X = ZeroPadding2D((3,3))(X_in)

		# Stage 1
		X = Conv2D(64 , (7,7) , strides = (2,2))(X)
		X = BatchNormalization(axis = 3)(X)
		X = Activation('relu')(X)
		X = MaxPooling2D((3,3) , strides = (2,2))(X)

		# Stage 2
		X = self.convolutional_block(X , f = 3 , filters = [64,64,256] , s = 1)
		X = self.identity_block(X ,  3, [64,64,256])
		X = self.identity_block(X ,  3, [64,64,256])

		# Stage 3
		X = self.convolutional_block(X , f = 3 , filters = [128,128, 512] , s = 2)
		X = self.identity_block(X , 3 , [128,128, 512])
		X = self.identity_block(X , 3 , [128,128, 512])
		X = self.identity_block(X , 3 , [128,128, 512])

		# Stage 4
		X = self.convolutional_block(X , f = 3 , filters = [256,256,1024] , s = 2)
		X = self.identity_block(X , 3 , [256,256,1024])
		X = self.identity_block(X , 3 , [256,256,1024])
		X = self.identity_block(X , 3 , [256,256,1024])
		X = self.identity_block(X , 3 , [256,256,1024])
		X = self.identity_block(X , 3 , [256,256,1024])

		# Stage 5
		X = self.convolutional_block(X , f=3 , filters = [512,512,2048] , s=2)
		X = self.identity_block(X , 3 , [512,512,2048])
		X = self.identity_block(X , 3 , [512,512,2048])

		X = AveragePooling2D((2,2) , name = 'avg_pool')(X)

		X = Flatten()(X)
		X = Dense(classes , activation = 'softmax' , name = 'fc'+str(classes) , kernel_initializer = glorot_uniform(seed=0))(X)

		model = Model(inputs = X_in , outputs = X , name = "Resnet50")

		return model


	def identity_block(self , X , f , filters) :

		# Retrieve the filters
		F1 , F2 , F3 = filters

		X_parallel = X

		# First
		X = Conv2D(filters = F1 , kernel_size = (1,1) , strides = (1,1) , padding = 'valid')(X)
		X = BatchNormalization(axis = 3)(X)
		X = Activation('relu')(X)

		# Second
		X = Conv2D(filters = F2 , kernel_size = (f,f) , strides = (1,1) , padding = 'same')(X)
		X = BatchNormalization(axis = 3)(X)
		X = Activation('relu')(X)

		# Third
		X = Conv2D(filters = F3 , kernel_size = (1,1) , strides = (1,1) , padding = 'valid')(X)
		X = BatchNormalization(axis = 3)(X)

		X = Add()([X , X_parallel])
		X = Activation('relu')(X)

		return X


	def convolutional_block(self , X , f , filters , s = 2 ) :

		# Retrieve the filters
		F1 , F2 , F3 = filters

		X_parallel = X

		# First
		X = Conv2D(filters = F1 , kernel_size = (1,1) , strides = (s,s) )(X)
		X = BatchNormalization(axis = 3)(X)
		X = Activation('relu')(X)

		# Second
		X = Conv2D(filters = F2 , kernel_size = (f,f) , strides = (1,1) , padding = 'same')(X)
		X = BatchNormalization(axis = 3)(X)
		X = Activation('relu')(X)

		# Third
		X = Conv2D(filters = F3 , kernel_size = (1,1) , strides = (1,1) , padding = 'valid')(X)
		X = BatchNormalization(axis = 3)(X)

		# For parallel connection
		X_parallel = Conv2D(filters = F3 , kernel_size = (1,1) , strides = (s,s) , padding = 'valid')(X_parallel)
		X_parallel = BatchNormalization(axis = 3)(X_parallel)

		X = Add()([X , X_parallel])
		X = Activation('relu')(X)

		return X

	def get_model() :
		return self.model


def main() :

	resnet = Resnet50()
	model = resnet.get_model()

	model.compile(optimizer = 'adam' , loss='categorical_crossentropy' , metrics=['accuracy'])
	print(model.summary())
	
	print("Success!")

if __name__ == '__main__':
	main()