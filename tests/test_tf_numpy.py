# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-12-07

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

################################################################################
precision = 64
tolerance = 1e-6
dtype_np = eval('np.float{}'.format(precision))
dtype_tf = eval('tf.complex{}'.format(precision*2))
image_file = './assets/lena.jpg'
im = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im, dsize=(200, 200))
im = im.astype(dtype_np)/255.  # note the np.float64

################################################################################
# NumPy

fft_numpy = np.fft.fft2(im)
fft_numpy = np.fft.fftshift(fft_numpy)

fft_numpy_mag_viz = np.log10(np.abs(fft_numpy))
fft_numpy_ang_viz = np.angle(fft_numpy)

ifft_numpy1 = np.fft.ifftshift(fft_numpy) 
ifft_numpy = np.fft.ifft2(ifft_numpy1)

################################################################################
# Tensorflow

im_tf = tf.convert_to_tensor(im[None,:,:],dtype_tf)  # add batch dim

fft_tf = tf.signal.fft2d(im_tf)
fft_tf = tf.signal.fftshift(fft_tf)

ifft_tf = tf.signal.ifftshift(fft_tf)
ifft_tf = tf.signal.ifft2d(ifft_tf)

ifft_tf_to_numpy = ifft_tf.numpy()
ifft_tf_to_numpy = np.squeeze(ifft_tf_to_numpy)
all_close_ifft = np.allclose(ifft_numpy, ifft_tf_to_numpy, atol=tolerance)
print('ifft all close: ', all_close_ifft)

fft_tf_to_numpy = np.squeeze(fft_tf.numpy())
ifft_tf_to_numpy  = np.squeeze(ifft_tf.numpy())

fft_tf_mag_viz = np.log10(np.abs(fft_tf_to_numpy))
fft_tf_ang_viz = np.angle(fft_tf_to_numpy)

################################################################################
# Tolerance checking

all_close_real = np.allclose(np.real(fft_numpy), np.real(fft_tf_to_numpy), atol=tolerance)
all_close_imag = np.allclose(np.imag(fft_numpy), np.imag(fft_tf_to_numpy), atol=tolerance)
print('fft allclose real: {}'.format(all_close_real))
print('fft allclose imag: {}'.format(all_close_imag))

all_close_real = np.allclose(np.real(ifft_numpy), np.real(ifft_tf_to_numpy), atol=tolerance)
all_close_imag = np.allclose(np.imag(ifft_numpy), np.imag(ifft_tf_to_numpy), atol=tolerance)
print('ifft allclose real: {}'.format(all_close_real))
print('ifft allclose imag: {}'.format(all_close_imag))

################################################################################

fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12,6))

# Plotting NumPy results
ax[0][0].imshow(im, cmap='gray')

ax[0][1].imshow(fft_numpy_mag_viz, cmap='gray')
ax[0][1].set_title('NumPy fft magnitude')
ax[0][2].imshow(fft_numpy_ang_viz, cmap='gray')
ax[0][2].set_title('NumPy fft spectrum')
ax[0][3].imshow(ifft_numpy.real, cmap='gray')
ax[0][3].set_title('NumPy ifft real')
ax[0][4].imshow(ifft_numpy.imag, cmap='gray')
ax[0][4].set_title('NumPy ifft imag')

# Plotting Tesnorflow results
ax[1][0].imshow(im, cmap='gray')
ax[1][1].imshow(fft_tf_mag_viz, cmap='gray')
ax[1][1].set_title('Tensorflow fft magnitude')
ax[1][2].imshow(fft_tf_ang_viz, cmap='gray')
ax[1][2].set_title('Tensorflow fft phase')
ax[1][3].imshow(ifft_tf_to_numpy.real, cmap='gray')
ax[1][3].set_title('Tensorflow ifft real')
ax[1][4].imshow(ifft_tf_to_numpy.imag, cmap='gray')
ax[1][4].set_title('Tensorflow ifft imag')

for cur_ax in ax.flatten():
    cur_ax.axis('off')
plt.tight_layout()
plt.show()
