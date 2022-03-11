# CS7CS4/CSU44061 Machine Learning
# Week 8 Assignment
# Boris Flesch (20300025)

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
import time
from itertools import cycle


############
# Part (i) #
############

# (i)(a)
def convolution(array, kernel):
	n = array.shape[0]
	k = kernel.shape[0]
	r = n - k + 1  # Output matrix size
	res = np.empty([r, r])

	for i in range(r):
		for j in range(r):
			tmp = 0
			for x in range(k):
				for y in range(k):
					tmp += array[x + i][y + j] * kernel[x][y]
			res[i][j] = tmp

	return res


def exec_part_i():
	# (i)(b)
	im = Image.open('image.jpg')
	rgb = np.array(im.convert('RGB'))
	r = rgb[:, :, 0]  # array of R pixels
	img_array = np.uint8(r)
	Image.fromarray(img_array).show()

	kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
	kernel1img = convolution(img_array, kernel1)
	Image.fromarray(kernel1img).show()

	kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])
	kernel2img = convolution(img_array, kernel2)
	Image.fromarray(kernel2img).show()

	kernel12img = convolution(img_array, kernel1)
	kernel12img = convolution(kernel12img, kernel2)
	Image.fromarray(kernel12img).show()

exec_part_i()


#############
# Part (ii) #
#############

def convnet(n=5000, L1_range=[0.0001], displayLoss=True, network='default', epochs=20):
	plt.rc('font', size=18)
	plt.rcParams['figure.constrained_layout.use'] = True

	lines = ["-", "--", "-.", ":"]
	linecycler = cycle(lines)

	for L1 in L1_range:

		# Model / data parameters
		num_classes = 10
		input_shape = (32, 32, 3)

		# the data, split between train and test sets
		(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

		x_train = x_train[1:n]; y_train=y_train[1:n]
		#x_test=x_test[1:500]; y_test=y_test[1:500]

		# Scale images to the [0, 1] range
		x_train = x_train.astype("float32") / 255
		x_test = x_test.astype("float32") / 255
		print("orig x_train shape:", x_train.shape)

		# convert class vectors to binary class matrices
		y_train_non_categorical = y_train
		y_test_non_categorical = y_test
		y_train = keras.utils.to_categorical(y_train, num_classes)
		y_test = keras.utils.to_categorical(y_test, num_classes)

		use_saved_model = False
		if use_saved_model:
			model = keras.models.load_model("cifar.model")
		else:
			model = keras.Sequential()

			if network == 'maxpooling':
				model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
				model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))
				model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
				model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
				model.add(MaxPooling2D(pool_size=(2, 2)))
			elif network == 'thinner_deeper':
				model.add(Conv2D(8, (3,3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
				model.add(Conv2D(8, (3,3), strides=(2,2), padding='same', activation='relu'))
				model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
				model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
				model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
				model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
			else:
				model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
				model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
				model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
				model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
			model.add(Dropout(0.5))
			model.add(Flatten())
			model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(L1)))
			model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
			model.summary()

			batch_size = 128

			start_time = time.time()
			history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

			print("Time to train the network:", time.time() - start_time)

			linestyle = next(linecycler)

			model.save("cifar.model")
			if displayLoss:
				plt.subplots_adjust(hspace=1)
				plt.subplot(211)
			plt.plot(history.history['accuracy'], label='train, L1=%.4f'%L1, linestyle=linestyle)
			plt.plot(history.history['val_accuracy'], label='val, L1=%.4f'%L1, linestyle=linestyle)
			plt.title('model accuracy (n = %d)'%n)
			plt.ylabel('accuracy')
			plt.xlabel('epoch')

			if displayLoss:
				plt.subplot(212)
				plt.plot(history.history['loss'], label='train, L1=%f'%L1, linestyle=linestyle)
				plt.plot(history.history['val_loss'], label='val, L1=%f'%L1, linestyle=linestyle)
				plt.title('model loss (n = %d)'%n)
				plt.ylabel('loss'); plt.xlabel('epoch')

	if displayLoss:
		plt.subplot(211)
		plt.legend(loc='upper left')
	else:
		plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

	if displayLoss:
		plt.subplot(212)
		plt.legend(loc='upper left')
	plt.show()

	preds = model.predict(x_train)
	y_pred = np.argmax(preds, axis=1)
	y_train1 = np.argmax(y_train, axis=1)
	print(classification_report(y_train1, y_pred))
	print(confusion_matrix(y_train1, y_pred))

	preds = model.predict(x_test)
	y_pred = np.argmax(preds, axis=1)
	y_test1 = np.argmax(y_test, axis=1)
	print(classification_report(y_test1, y_pred))
	print(confusion_matrix(y_test1, y_pred))

	# Compare this performance against a simple baseline e.g. always predicting the most common label.
	x_train_flat = [];
	for i in range(x_train.shape[0]):
		x_train_flat.append(x_train[i].flatten(order='C'))
	x_train_flat = np.array(x_train_flat);

	x_test_flat = [];
	for i in range(x_test.shape[0]):
		x_test_flat.append(x_test[i].flatten(order='C'))
	x_test_flat = np.array(x_test_flat);

	dummy_clf = DummyClassifier(strategy="most_frequent")
	dummy_clf.fit(x_train_flat, y_train_non_categorical)
	print("Baseline score:", dummy_clf.score(x_test_flat, y_test_non_categorical))


# (ii)(b)(ii)
convnet()

# (ii)(b)(iii)
convnet(n=5000)
convnet(n=10000)
convnet(n=20000)
convnet(n=40000)

# (ii)(b)(iv)
convnet(n=5000, L1_range=[0, 0.0001, 0.01, 1, 100], displayLoss=False)
convnet(n=40000, L1_range=[0, 0.0001, 0.001, 0.01], displayLoss=False)

# (ii)(c)(i)
convnet(n=40000, network='maxpooling')

# (ii)(d) (optional)
convnet(network='thinner_deeper')
convnet(epochs=100, network='thinner_deeper')
convnet(epochs=100, n=40000, network='thinner_deeper')
convnet(epochs=45, n=40000, network='thinner_deeper')
