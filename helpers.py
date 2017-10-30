import os
from keras.preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import numpy as np
import cv2

alphabet = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11,
            'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22,
            'x': 23, 'y': 24, 'z': 25}


# Needed for transformation of character to number.
def char_to_num(char):
    num = alphabet[char]
    return num


# Needed for transformation of number to character.
def num_to_char(num):
    for key in alphabet:
        if alphabet[key] == num:
            return key

# Load file paths and labels them.
def load_chars74k_data(dir="chars74k-lite"):
    filenames = []
    label_list = []

    for path, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.jpg'):
                file = path + '/' + file
                filenames.append(file)

                label = path[-1:]
                label_list.append(label)
    return filenames, label_list

# Creates the dataset.
def create_dataset(file_paths, label_set, with_denoising=False):
    data_x = []
    data_y = []

    for path in file_paths:
        single_x = np.asarray(PIL.Image.open(path)).flatten()

        # Denoise image with help of OpenCV (increase time of computing).
        if with_denoising:
            single_x = cv2.fastNlMeansDenoising(single_x).flatten()
        data_x.append(single_x)

    for l in label_set:
        l_to_num = char_to_num(l)
        data_y.append(l_to_num)

    np_data_x = np.array(data_x)
    np_data_y = np.array(data_y)
    return np_data_x, np_data_y


# Use the Keras data generator to augment data.
def create_datagenerator(x_train, x_test, y_train, y_test):
    train_datagen = ImageDataGenerator(
        rescale= 1. / 255,
        rotation_range= 0. / 180,
        vertical_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow(x=x_train, y=y_train)
    validation_generator = test_datagen.flow(x=x_test, y=y_test)

    return train_generator, validation_generator