import numpy as np
import cv2

import keras  # broken for keras >= 2.0, use 1.2.2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape


def load_weights(model, weight_file):
    data = np.fromfile(weight_file, np.float32)
    data = data[4:]

    index = 0
    for layer in model.layers:
        shape = [w.shape for w in layer.get_weights()]
        if shape != []:
            kshape, bshape = shape
            bia = data[index:index + np.prod(bshape)].reshape(bshape)
            index += np.prod(bshape)
            ker = data[index:index + np.prod(kshape)].reshape(kshape)
            index += np.prod(kshape)
            layer.set_weights([ker, bia])


def preprocess(img):
    # Crops Image to only include relevant parts of picture
    cropped = img[300:650, 500:, :]
    # Resizes image to  fit input. Model requires 448x448
    resized = cv2.resize(cropped, (448, 448))
    # Simple image normalization (-1,1)
    normalized = 2.0 * resized / 255.0 - 1
    # Model works on dimensions (channel, height, width) thus image array has to be transposed
    transposed = np.transpose(normalized, (2, 0, 1))
    return transposed


import matplotlib.image as mpimg

test_image = mpimg.imread('test_images/test1.jpg')
pre_processed = preprocess(test_image)
#expand dimensions because input expects
batch = np.expand_dims(pre_processed, axis=0)
print(batch.shape)
# batch_output = model.predict(batch)
# print(batch_output.shape)