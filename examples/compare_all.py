# MIT License
#
# Copyright (c) 2020 Didan Deng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Didan Deng
# Date Created: 2020-03-31

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import cv2
import tensorflow as tf

from steerable.SCFpyr_NumPy import SCFpyr_NumPy
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from steerable.SCFpyr_TF import SCFpyr_TF
import steerable.utils as utils

################################################################################
################################################################################
# Common
precision=64
image_file = './assets/lena.jpg'
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (200,200))
image = image.astype(eval('np.float{}'.format(precision)))

# Number of pyramid levels
pyr_height = 5

# Number of orientation bands
pyr_nbands = 4

# Tolerance for error checking
tolerance = 1e-6

################################################################################
# NumPy

pyr_numpy = SCFpyr_NumPy(pyr_height, pyr_nbands, scale_factor=2, precision=precision)
coeff_numpy = pyr_numpy.build(image)
reconstruction_numpy = pyr_numpy.reconstruct(coeff_numpy)
reconstruction_numpy = reconstruction_numpy.astype(np.uint8)

print('#'*60)

################################################################################
# PyTorch

device = torch.device('cuda:0')

im_batch = torch.from_numpy(image[None,None,:,:])
im_batch = im_batch.to(device)
pyr_torch = SCFpyr_PyTorch(pyr_height, pyr_nbands, device=device, precision=precision)
coeff_torch = pyr_torch.build(im_batch)
reconstruction_torch = pyr_torch.reconstruct(coeff_torch)
reconstruction_torch = reconstruction_torch.cpu().numpy()[0,]

coeff_torch = utils.extract_from_batch(coeff_torch, 0)

################################################################################
# Tensorflow
dtype_tf = eval('tf.complex{}'.format(precision*2))
im_batch = tf.convert_to_tensor(image[None,:,:, None], dtype_tf) # N, W, H, C


pyr_tf = SCFpyr_TF(pyr_height, pyr_nbands, precision=precision)
coeff_tf = pyr_tf.build(im_batch)
reconstruction_tf = pyr_tf.reconstruct(coeff_tf)
reconstruction_tf = reconstruction_tf.numpy()[0,]

# Extract first example from the batch and move to CPU
coeff_tf = utils.extract_from_batch(coeff_tf, 0)

################################################################################
# Check correctness

print('#'*60)
assert len(coeff_numpy) == len(coeff_torch)

for level, _ in enumerate(coeff_numpy):

    print('Pyramid Level {level}'.format(level=level))
    coeff_level_numpy = coeff_numpy[level]
    coeff_level_torch = coeff_torch[level]
    coeff_level_tf = coeff_tf[level]

    assert isinstance(coeff_level_torch, type(coeff_level_numpy))
    assert isinstance(coeff_level_tf, type(coeff_level_numpy))
    
    if isinstance(coeff_level_numpy, np.ndarray):

        # Low- or High-Pass
        print('  NumPy.   min = {min:.3f}, max = {max:.3f},'
              ' mean = {mean:.3f}, std = {std:.3f}'.format(
                  min=np.min(coeff_level_numpy), max=np.max(coeff_level_numpy), 
                  mean=np.mean(coeff_level_numpy), std=np.std(coeff_level_numpy)))

        print('  PyTorch. min = {min:.3f}, max = {max:.3f},'
              ' mean = {mean:.3f}, std = {std:.3f}'.format(
                  min=np.min(coeff_level_torch), max=np.max(coeff_level_torch), 
                  mean=np.mean(coeff_level_torch), std=np.std(coeff_level_torch)))
        print('  Tensorflow. min = {min:.3f}, max = {max:.3f},'
              ' mean = {mean:.3f}, std = {std:.3f}'.format(
                  min=np.min(coeff_level_tf), max=np.max(coeff_level_tf), 
                  mean=np.mean(coeff_level_tf), std=np.std(coeff_level_tf)))


        # Check numerical correctness
        assert np.allclose(coeff_level_numpy, coeff_level_torch, atol=tolerance)
        assert np.allclose(coeff_level_numpy, coeff_level_tf, atol=tolerance)
    elif isinstance(coeff_level_numpy, list):

        # Intermediate bands
        for band, _ in enumerate(coeff_level_numpy):

            band_numpy = coeff_level_numpy[band]
            band_torch = coeff_level_torch[band]
            band_tf = coeff_level_tf[band]

            print('  Orientation Band {}'.format(band))
            print('    NumPy.   min = {min:.3f}, max = {max:.3f},'
                  ' mean = {mean:.3f}, std = {std:.3f}'.format(
                      min=np.min(band_numpy), max=np.max(band_numpy), 
                      mean=np.mean(band_numpy), std=np.std(band_numpy)))

            print('    PyTorch. min = {min:.3f}, max = {max:.3f},'
                  ' mean = {mean:.3f}, std = {std:.3f}'.format(
                      min=np.min(band_torch), max=np.max(band_torch), 
                      mean=np.mean(band_torch), std=np.std(band_torch)))
            print('    Tensorflow. min = {min:.3f}, max = {max:.3f},'
                  ' mean = {mean:.3f}, std = {std:.3f}'.format(
                      min=np.min(band_tf), max=np.max(band_tf), 
                      mean=np.mean(band_tf), std=np.std(band_tf)))

            # Check numerical correctness
            assert np.allclose(band_numpy, band_torch, atol=tolerance)
            assert np.allclose(band_numpy, band_tf, atol=tolerance)

################################################################################
# Visualize

assert np.allclose(reconstruction_numpy.astype(np.uint8), reconstruction_torch.astype(np.uint8), atol=1)
assert np.allclose(reconstruction_numpy.astype(np.uint8), reconstruction_tf.astype(np.uint8), atol=1)
coeff_grid_numpy = utils.make_grid_coeff(coeff_numpy, normalize=False)
coeff_grid_torch = utils.make_grid_coeff(coeff_torch, normalize=False)
coeff_grid_tf = utils.make_grid_coeff(coeff_tf, normalize=False)
cv2.imshow('image', image.astype(np.uint8))
cv2.imshow('coeff numpy', np.ascontiguousarray(coeff_grid_numpy))
cv2.imshow('coeff torch', np.ascontiguousarray(coeff_grid_torch))
cv2.imshow('coeff tf', np.ascontiguousarray(coeff_grid_tf))
cv2.imshow('reconstruction numpy', reconstruction_numpy.astype(np.uint8))
cv2.imshow('reconstruction torch', reconstruction_torch.astype(np.uint8))
cv2.imshow('reconstruction tf', reconstruction_tf.astype(np.uint8))
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
