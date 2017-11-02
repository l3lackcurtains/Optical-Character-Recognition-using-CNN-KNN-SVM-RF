import cv2
import helpers
import image_detection as detector
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from collections import Counter
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split

batch_size = 64
num_classes = 26
epochs = 30
img_rows, img_cols = 20, 20

print('Start loading data.')
files, labels = helpers.load_chars74k_data()
X, y = helpers.create_dataset(files, labels)
print('Data has been loaded.')

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=2, train_size=0.8)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

train_generator, validation_generator = helpers.create_datagenerator(x_train, x_test, y_train, y_test)


print('\n***** Recording time *****')
e1 = cv2.getTickCount()

# Convolutional network with Keras.
print('Start training the model.')

'''
The Sequential model is a linear stack of layers.
The first layer in a Sequential model needs to receive information about its input shape.
'''
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

'''
Before training a model, you need to configure the learning process, which is done via the compile method.
Optimization is the process of finding the set of parameters WW that minimize the loss function.
the loss function lets us quantify the quality of any particular set of weights W
'''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=10000 // batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps=6000 // batch_size)

# Calculate loss and accuracy.
score = model.evaluate(x_test, y_test, verbose=0)
print('Model has been trained.')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

e2 = cv2.getTickCount()
time0 = (e2 - e1) / cv2.getTickFrequency()
print('\n ***** Total time elapsed:',time0, ' *****')

# Predict on detection-1.jpg
detection1 = './detection-images/detection-1.jpg'
samples1 = detector.sliding_window(detection1)

samples_tf1 = samples1.reshape(samples1.shape[0], 20, 20, 1)
samples_tf1 = samples_tf1.astype('float32')

print('Start detection example image: ', detection1)
predictions1 = model.predict(samples_tf1)
value_list1 = []

for pred in predictions1:
    i = 0
    for value in pred:
        if value > 0.9:
            value_list1.append(helpers.num_to_char(i))
        i += 1

predictCount = Counter(value_list1)
print('\nPrediction result', predictCount)

print('\nResults in Probability\n')
for k, v in predictCount.items():
	print(k, ':', v/len(predictions1))

out = max(value_list1,key=value_list1.count)
print('\nMost Predicted Character is', out)

resImg = ''
if out == 'e':
	resImg = './detection-images/detection-5.jpg'
if out == 'a':
	resImg = './detection-images/detection-4.jpg'

if resImg:
	img = cv2.imread(resImg,0)
	cv2.imshow("Detected Image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
