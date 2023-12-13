

import numpy as np # linear algebra


from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from skimage import color
import keras as keras
import tensorflow as tf
import glob
import cv2 as cv2
import os
import pdb
folder_path='C:/Users/Morteza/Desktop/YouTube/coding/Image Colorization/BW_small/' 
images1 = []
for img in os.listdir(folder_path):
    img=folder_path+img
    img = load_img(img, target_size=(100,100)) 
    img = img_to_array(img)/ 255
    X= color.rgb2gray(img)
    images1.append(X)
#pdb.set_trace()

folder_path='C:/Users/Morteza/Desktop/YouTube/coding/Image Colorization/colored_small/' 
images2 = []
for img in os.listdir(folder_path):
    #print(folder_path+img)
    img=folder_path+img
    img = load_img(img, target_size=(100,100)) 
    img = img_to_array(img)/ 255
    lab_image = rgb2lab(img)
    lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]
    # The input will be the black and white layer
    Y = lab_image_norm[:,:,1:]

    images2.append(Y)
#pdb.set_trace()


X = np.array(images1)
Y = np.array(images2)
#pdb.set_trace()

x1 = keras.Input(shape=(None, None, 1))

x2 = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(x1)
x3 = Conv2D(16, (3, 3), activation='relu', padding='same')(x2)
x4 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x3)
x5 = Conv2D(32, (3, 3), activation='relu', padding='same')(x4)
x6 = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x5)
x7 = UpSampling2D((2, 2))(x6)
x8 = Conv2D(32, (3, 3), activation='relu', padding='same')(x7)
x9 = UpSampling2D((2, 2))(x8)
x10 = Conv2D(16, (3, 3), activation='relu', padding='same')(x9)
x11 = UpSampling2D((2, 2))(x10)
x12 = Conv2D(2, (3,3), activation='sigmoid', padding='same')(x11)

x12=tf.reshape(x12,(104,104,2))
x12 = tf.image.resize(x12,[100, 100])
x12=tf.reshape(x12,(1,100, 100,2))

# Finish model
model = keras.Model(x1, x12)

model.compile(optimizer='rmsprop', loss='mse')
model.fit(X,Y, batch_size=1, epochs=400, verbose=1)

model.evaluate(X, Y, batch_size=1)
#pdb.set_trace()

folder_path='C:/Users/Morteza/Desktop/YouTube/coding/Image Colorization/Test/' 
img='Gray_33.jpg'
img=folder_path+img


img = load_img(img, target_size=(100,100),color_mode = "grayscale") 
img = img_to_array(img)/ 255
ss=img.shape

X = np.array(img)
X = np.expand_dims(X, axis=2)
X=np.reshape(X,(1,100,100,1))
output = model.predict(X)
output=np.reshape(output,(100,100,2))
output=cv2.resize(output,(ss[1],ss[0]))
AB_img = output
outputLAB = np.zeros((ss[0],ss[1], 3))
img=np.reshape(img,(100,100))
outputLAB[:,:,0]=img
outputLAB[:,:,1:]=AB_img
outputLAB = (outputLAB * [100, 255, 255]) - [0, 128, 128]
rgb_image = lab2rgb(outputLAB)

import matplotlib.pyplot as plt

imshow(rgb_image)
plt.show()
