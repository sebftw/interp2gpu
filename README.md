# Fast Spline Interpolation using GPU Acceleration
interp2gpu is a drop-in replacement to [interp2](https://mathworks.com/help/matlab/ref/interp2.html) to perform spline interpolation on the GPU in MATLAB.

## How to use
Call it as `Vq = interp2gpu(V, Xq, Yq, "spline")`.  
For an explanation of inputs, see  [https://mathworks.com/help/matlab/ref/interp2.html](https://mathworks.com/help/matlab/ref/interp2.html) from Mathworks.


## Features
:heavy_check_mark: Multiple images can be processed with one call to interp2gpu.  
:heavy_check_mark: The method "spline_approx" performs fast approximated spline interpolation.  
:x: Does not currently support double-precision inputs.  
:x: Does not currently support arbitrary input pixel grids.  

You can add support for double-precision inputs and arbitrary input pixel grids fairly easily - feel free to contribute.

## Files
- `example.m` is an example that shows how to use interp2gpu.
- `interp2gpu.m` is the main function.
- `gpuThomas2D.m` is used to compute the image derivatives required to evaluate the spline.
- `get_kernel.m` is used to cache CUDA kernels, so they do not have to be loaded every time.
- `kernels/getInterpolation2D.cu` contains the CUDA C++ source code used to evaluate the spline.
- `kernels/compile_kernels.m` is used to compile the CUDA code.
- `kernels/getInterpolation2D.ptx` is the compiled CUDA kernel.


