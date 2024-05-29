#!usr/bin/env python

img = '../data/cat.jpg'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# get the image flip in  numpy 
flp_1 = np.flip(img)

shape = [28, 28, 3]

# create a tensor place holder fo the image 
x = tf.placeholder(tf.float32, shape=shape)
flip_2 = tf.image.flip_left_right(x)
flip_3 = tf.image.flip_up_down(x)
flip_4 = tf.image.random_flip_left_right(x)
flip5 = tf.image.random_flip_up_down(x)


plt.imshow(flp_1)
plt.imshow(flip_2)
plt.imshow(flip_3)
plt.imshow(flip_4)
plt.imshow(flip5)
