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

import argparse
import cv2

from steerable.SCFpyr_NumPy import SCFpyr_NumPy
import steerable.utils as utils
import numpy as np

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

    image = cv2.imread('./assets/lena.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(200,200))

    coeff = pyr.build(image.astype(dtype_np))

    grid = utils.make_grid_coeff(coeff, normalize=True)

    cv2.imwrite('./assets/coeff.png', grid)
    cv2.imshow('image', image)
    cv2.imshow('coeff', grid)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

    
