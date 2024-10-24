# Fast Spline Interpolation using GPU Acceleration
interp2gpu is a drop-in replacement to [interp2](https://mathworks.com/help/matlab/ref/interp2.html) to perform spline interpolation on the GPU.

## How to use
Call it as `Vq = interp2gpu(V, Xq, Yq, "spline")`.  
For an explanation of inputs, see  [https://mathworks.com/help/matlab/ref/interp2.html](https://mathworks.com/help/matlab/ref/interp2.html) from Mathworks.


## Features and limitations 
:heavy_check_mark: Multiple images can be processed in one call to interp2gpu.  
:heavy_check_mark: The method "spline_approx" performs fast approximated spline interpolation.  
:x: Does not currently support double-precision inputs.  
:x: Does not currently support arbitrary input pixel grids.  

Support for double-precision inputs and arbitrary input pixel grids can be added relatively easily - feel free to contribute.
