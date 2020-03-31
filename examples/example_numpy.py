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

import time
import argparse
import numpy as np

from steerable.SCFpyr_NumPy import SCFpyr_NumPy
import steerable.utils as utils

################################################################################
################################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, default='./assets/patagonia.jpg')
    parser.add_argument('--batch_size', type=int, default='1')
    parser.add_argument('--image_size', type=int, default='200')
    parser.add_argument('--pyr_nlevels', type=int, default='5')
    parser.add_argument('--pyr_nbands', type=int, default='4')
    parser.add_argument('--pyr_scale_factor', type=int, default='2')
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--precision', type=int, default=32, choices=[32, 64])
    config = parser.parse_args()
    dtype_np = eval('np.float{}'.format(config.precision))

    ############################################################################
    # Build the complex steerable pyramid

    pyr = SCFpyr_NumPy(
        height=config.pyr_nlevels, 
        nbands=config.pyr_nbands,
        scale_factor=config.pyr_scale_factor,  
        precision = config.precision
    )

    ############################################################################
    # Create a batch and feed-forward

    start_time = time.time()

    im_batch_numpy = utils.load_image_batch(config.image_file, config.batch_size, config.image_size)
    im_batch_numpy = im_batch_numpy.astype(dtype_np).squeeze(1)  # no channel dim for NumPy

    # Compute Steerable Pyramid
    start_time = time.time()
    for image in im_batch_numpy:
        coeff = pyr.build(image)

    duration = time.time()-start_time
    print('Finishing decomposing {batch_size} images in {duration:.1f} seconds.'.format(
        batch_size=config.batch_size,
        duration=duration
    ))

    ############################################################################
    # Visualization

    if config.visualize:
        import cv2
        coeff_grid = utils.make_grid_coeff(coeff, normalize=True)
        cv2.imshow('image', (im_batch_numpy[0,]*255.).astype(np.uint8))
        cv2.imshow('coeff', coeff_grid)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
