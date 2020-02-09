# importing libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.optimizers import RMSprop



img_width, img_height = 224, 224

train_data_dir = 'Database/tranig'
validation_data_dir = 'Database/test'
nb_train_samples = 403
nb_validation_samples = 100
epochs = 10
batch_size = 2
activation = 'softsign'
if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape = input_shape))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(64, (5, 5)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('softsign'))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Conv2D(512, (2, 2)))
model.add(Activation(activation))
model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('softmax'))
model.add(Dense(512))
model.add(Activation('softmax'))
model.add(Dropout(0.8))
model.add(Dense(8, activation='softmax'))

sgd = RMSprop(lr=0.0001,rho=0.9, epsilon=None, decay=0.0)

model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#model.compile(loss ='binary_crossentropy',
#					optimizer ='rmsprop',
#				metrics =['accuracy'])

train_datagen = ImageDataGenerator(
				rescale = 1. / 255,
				shear_range = 0.2,
				zoom_range = 0.2,
			horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
							target_size =(img_width, img_height),
					batch_size = batch_size, class_mode ='binary')

validation_generator = test_datagen.flow_from_directory(
									validation_data_dir,
				target_size =(img_width, img_height),
		batch_size = batch_size, class_mode ='binary')

model.fit_generator(train_generator,
	steps_per_epoch = nb_train_samples // batch_size,
	epochs = epochs, validation_data = validation_generator,
	validation_steps = nb_validation_samples // batch_size)


model.save('money1.h5')
