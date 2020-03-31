# Complex Steerable Pyramid in PyTorch, Tensorflow and Numpy

This is an implementation of the Complex Steerable Pyramid described in [Portilla and Simoncelli (IJCV, 2000)](http://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Portilla99), forked from [tomrunia/PyTorchSteerablePyramid](https://github.com/tomrunia/PyTorchSteerablePyramid)

The complex steerable pyramid expects a batch of images of shape (`[N,C,H,W]` for Pytorch and `[N, W, H, C]` for Tensorflow) with current support only for grayscale images (`C=1`). It returns a `list` structure containing the low-pass, high-pass and intermediate levels of the pyramid for each image in the batch (as `torch.Tensor` and `tf.Tensor`). Computing the steerable pyramid is significantly faster on the GPU as can be observed from the runtime benchmark below. 

<a href="/assets/coeff.png"><img src="/assets/coeff.png" width="700px" ></a>

## Usage

Please check the scripts in examples/

## Consistnecy

`test_tf_numpy.py` and `test_tf_torch.py` in tests/

## Benchmark

Performing parallel the CSP decomposition on the GPU results in a significant speed-up. Increasing the batch size will give faster runtimes. The plot below shows a comprison between the `numpy`, `torch` and `tensorflow` implementations as function of the batch size `N` and input signal length. These results were obtained on a powerful Linux desktop with NVIDIA GTX1080Ti GPU.

<a href="/assets/runtime_benchmark.pdf"><img src="/assets/runtime_benchmark.png" width="700px" ></a>

## Installation

Clone and install:

```sh
https://github.com/wtomin/SteerablePyramid_Torch_TF_Numpy.git
cd SteerablePyramid_Torch_TF_Numpy
pip install -r requirements.txt
python setup.py install
```

## Requirements

- Python 2.7 or 3.6 (other versions might also work)
- Numpy (developed with 1.15.4)
- PyTorch >= 0.4.0 (developed with 1.0.0)
- Tensorflow >= 2.1.0 


## References

- [J. Portilla and E.P. Simoncelli, Complex Steerable Pyramid (IJCV, 2000)](http://www.cns.nyu.edu/pub/eero/portilla99-reprint.pdf)
- [The Steerable Pyramid](http://www.cns.nyu.edu/~eero/steerpyr/)
- [Official implementation: matPyrTools](http://www.cns.nyu.edu/~lcv/software.php)
- [perceptual repository by Dzung Nguyen](https://github.com/andreydung/Steerable-filter)

## License

MIT License

Copyright (c) 2018 Didan Deng (dengdidanwtomin@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
