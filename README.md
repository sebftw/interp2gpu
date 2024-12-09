# Fast 2D Spline Interpolation
__interp2gpu__ is a drop-in replacement to `interp2` to perform fast spline interpolation on the GPU in MATLAB.

## How to use
Call it as `Vq = interp2gpu(V, Xq, Yq, "spline")`.  
For an explanation of inputs, see  [https://mathworks.com/help/matlab/ref/interp2.html](https://mathworks.com/help/matlab/ref/interp2.html) from Mathworks.  
An example of using interp2gpu is given in [example.m](example.m).

## Installation
You can download the code with [this link](https://github.com/sebftw/interp2gpu/archive/refs/heads/main.zip).  
Then run `addpath(CODEPATH)`, where CODEPATH is the path of the directory where interp2gpu is located.  
You can now use interp2gpu!

## Features
:heavy_check_mark: Multiple images can be processed with one call to interp2gpu.  
:heavy_check_mark: The method "spline_approx" performs fast approximated spline interpolation.  
:x: Does not currently support double-precision inputs.  
:x: Does not currently support arbitrary input pixel grids.  

You can add support for double-precision inputs and non-uniform pixel grids relatively easily - feel free to contribute.

## Files
- `example.m` is an example that shows how to use interp2gpu.
- `interp2gpu.m` is the main function.
- `gpuThomas2D.m` is used to compute the image derivatives required to evaluate the spline.
- `get_kernel.m` is used to cache CUDA kernels, so they do not have to be loaded every time.
- `Spline_paper.pdf` is a paper containing the theory of the method and its application in ultrasound imaging..
- `kernels/getInterpolation2D.cu` contains the CUDA C++ source code used to evaluate the spline.
- `kernels/compile_kernels.m` is used to compile the CUDA code.
- `kernels/getInterpolation2D.ptx` is the compiled CUDA kernel.


