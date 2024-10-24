#include <cuda_runtime.h>
#include "helper_math.h"
#define vsub(C, X, Y)   v[(X)+bounds.x*(Y)+bounds.x*bounds.y*(C-1)+offset]
#define vsub_split(C, X, Y)   make_float2(v_real[(X)+bounds.x*(Y)+bounds.x*bounds.y*(C-1)+offset], v_imag[(X)+bounds.x*(Y)+bounds.x*bounds.y*(C-1)+offset])
#define fsub(X, Y)   f[(X)+bounds.x*(Y)+offset]

// We can also define vsub with the four derivative values in the leading
// dimension, as 
// #define vsub(C, X, Y)   v[(C-1)+4*(X)+4*bounds.x*(Y)+offset]
// This will lead to faster interpolation due to better cache use, but if
// only one interpolation is done, the speed-up is not worth it due to the
// time to do the one-time computation:
// derivatives = permute(derivatives, [3, 1, 2]);
// Additionally one can then do vectorized loads, or float4 textures.

#define SMALL_ANGLE 1e-6f // 1e-6f, 1.2e-7f, 1e-30

template <typename Typointer, typename Ty2>
class split_complex {
public:
    Typointer real;
    Typointer imag;
    
    __host__ __device__
    split_complex(Typointer real, Typointer imag) : real(real), imag(imag) {}
    
    // Retrieve the i'th element, as if this was a normal array.
    template<typename Tidx>
    __host__ __device__ inline
    Ty2 operator [] (Tidx i) const {return {real[i], imag[i]};}

   template<typename Tidx>
    __host__ __device__ inline
    split_complex operator + (Tidx i) const {return split_complex(real+i, imag+i);}
    // ^ What about the Ty2 type? Is that inferred simply from scope?
};

template <typename Ty, typename Tv, typename Tdv>
__device__ Tv do_interp2d_far_split(const Tv * __restrict__ v, Tdv dvdx, Tdv dvdy, Tdv dvdxdy, const int2 p, const Ty x, const Ty y, const int2 bounds) {
    int idx00 = p.y * bounds.x + p.x;
    int idx10 = idx00+1;  // FIXME: Maybe 10 and 01 should be swapped!
    int idx01 = idx00+bounds.x;
    int idx11 = idx00+bounds.x+1;
    auto    f00   = v[idx00],
            fx00  = dvdx[idx00],
            fy00  = dvdy[idx00],
            fxy00 = dvdxdy[idx00],
        
            f01   = v[idx01],
            fx01  = dvdx[idx01],
            fy01  = dvdy[idx01],
            fxy01 = dvdxdy[idx01],
        
            f10   = v[idx10],
            fx10  = dvdx[idx10],
            fy10  = dvdy[idx10],
            fxy10 = dvdxdy[idx10],
        
            f11   = v[idx11],
            fx11  = dvdx[idx11],
            fy11  = dvdy[idx11],
            fxy11 = dvdxdy[idx11];
    
    auto y2 = y * y;
    auto x2 = x * x;  // another form without squaring may be nice, if at all possible.
    return ((-3 + 3 * y) * y2 + (2 - 2 * y) * y2 * x) * x2 * fy11 + ((3 + (-6 + 3 * y) * y) * y + (-2 + (4 - 2 * y) * y) * y * x) * x2 * fy10 + ((-1 + y) * y2 + ((3 - 3 * y) * y2 + (-2 + 2 * y) * y2 * x) * x2) * fy01 + ((1 + (-2 + y) * y) * y + ((-3 + (6 - 3 * y) * y) * y + (2 + (-4 + 2 * y) * y) * y * x) * x2) * fy00 + ((1 - y) * y2 + (-1 + y) * y2 * x) * x2 * fxy11 + ((-1 + (2 - y) * y) * y + (1 + (-2 + y) * y) * y * x) * x2 * fxy10 + ((-1 + y) * y2 + ((2 - 2 * y) * y2 + (-1 + y) * y2 * x) * x) * x * fxy01 + ((1 + (-2 + y) * y) * y + ((-2 + (4 - 2 * y) * y) * y + (1 + (-2 + y) * y) * y * x) * x) * x * fxy00 + ((-3 + 2 * y) * y2 + (3 - 2 * y) * y2 * x) * x2 * fx11 + (-1 + (3 - 2 * y) * y2 + (1 + (-3 + 2 * y) * y2) * x) * x2 * fx10 + ((3 - 2 * y) * y2 + ((-6 + 4 * y) * y2 + (3 - 2 * y) * y2 * x) * x) * x * fx01 + (1 + (-3 + 2 * y) * y2 + (-2 + (6 - 4 * y) * y2 + (1 + (-3 + 2 * y) * y2) * x) * x) * x * fx00 + ((9 - 6 * y) * y2 + (-6 + 4 * y) * y2 * x) * x2 * f11 + (3 + (-9 + 6 * y) * y2 + (-2 + (6 - 4 * y) * y2) * x) * x2 * f10 + ((3 - 2 * y) * y2 + ((-9 + 6 * y) * y2 + (6 - 4 * y) * y2 * x) * x2) * f01 + (1 + (-3 + 2 * y) * y2 + (-3 + (9 - 6 * y) * y2 + (2 + (-6 + 4 * y) * y2) * x) * x2) * f00;
}

template <typename T, typename Tv, typename Tdv>
__device__ Tv getP_far_split(const Tv * __restrict__ v, Tdv dvdx, Tdv dvdy, Tdv dvdxdy, T& qx, T& qy, int2 &bounds, Tv extrapval) {
    // Terribly unreadable function. :(
    
    // Matlab uses 1-indexing
    int2 p = make_int2(floorf(qx), floorf(qy));  // FIXME: Use floor not floorf
    
    p.x = p.x - (p.x>=bounds.x-1 ? 1 : 0);  // In case we are exactly on the image edge, take one step back (use x=1 instead of x=0 in this edge case).
    p.y = p.y - (p.y>=bounds.y-1 ? 1 : 0);
    if ((qx < 0) || (qy < 0) || (qx > bounds.x-1) || (qy > bounds.y-1)) {
        // Extrapolation outside of image: Set to zero.
        // Note that the condition above should probably have been qx >= bounds.x-1 || (qy >= bounds.y-1)
        //  but we replicate this minor "error", to match the behaviour of MATLAB.
        return extrapval;
    }
    
    // Move the query to relative inside interval: [0, 1]
    qx -= p.x;
    qy -= p.y;
    
    Tv q = do_interp2d_far_split(v, dvdx, dvdy, dvdxdy, p, qx, qy, bounds);

    // Store the offset position of the interval in the bounds variable.
    bounds.x = p.x;
    bounds.y = p.y;
    return q;
}

// Define type mapping for vectorized types float->float2, double->double2, etc.
// https://stackoverflow.com/a/17834484
template<typename T> struct Vectypes{};
template<> struct Vectypes<float>{typedef float2 type; };
template<> struct Vectypes<double>{typedef double2 type; };


template <typename T, typename Tv, typename Tdv>
__device__ void getInterpolation2D_far_generic(Tv * __restrict__ q,
                                             const Tv * __restrict__ v,
                                             Tdv dvdx,
                                             Tdv dvdy, 
                                             Tdv dvdxdy,
                                             const int xs, const int ys, const T * __restrict__ xq, const T * __restrict__ yq, const int xs_out, const int ys_out, const Tv extrapval)  {
    // This is the function that we currently use for 2D interpolation.
    // First offset to get the page this block must work on.
    int batch_idx = blockIdx.z;
    int page = batch_idx * xs * ys;
    int page_out = batch_idx * xs_out * ys_out;
    yq += page_out;
    xq += page_out;
    q += page_out;

    v += page;
    dvdx = dvdx + page;
    dvdy = dvdy + page;
    dvdxdy = dvdxdy + page;

    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ys_out; j += blockDim.y * gridDim.y)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < xs_out; i += blockDim.x * gridDim.x) {

            int2 bounds = make_int2(xs, ys);
            int idx = i + xs_out * j;
            T query_x = xq[idx]-1;
            T query_y = yq[idx]-1;
            q[idx] = getP_far_split<T, Tv, Tdv>(v, dvdx, dvdy, dvdxdy, query_x, query_y, bounds, extrapval);
        }
}


// Template instantiation begins here.

// Far split case (far because the derivatives are stored in separate arrays, split because the complex and imaginary parts are split into separate arrays).
__global__ void getInterpolation2D_far_split(float2 * __restrict__ q,
                                             const float2 * __restrict__ v,
                                             const float * __restrict__ dvdx_real, const float * __restrict__ dvdx_imag,
                                             const float * __restrict__ dvdy_real, const float * __restrict__ dvdy_imag,
                                             const float * __restrict__ dvdxdy_real, const float * __restrict__ dvdxdy_imag,
                                             const int xs, const int ys, const float * __restrict__ xq, const float * __restrict__ yq, const int xs_out, const int ys_out, const float2 extrapval)  {
    typedef float Tdv;
    typedef typename Vectypes<Tdv>::type Tvec;  // https://stackoverflow.com/a/17834484
    auto dvdx = split_complex<const Tdv * __restrict__, Tvec>(dvdx_real, dvdx_imag);
    auto dvdy = split_complex<const Tdv * __restrict__, Tvec>(dvdy_real, dvdy_imag);
    auto dvdxdy = split_complex<const Tdv * __restrict__, Tvec>(dvdxdy_real, dvdxdy_imag);
    
    getInterpolation2D_far_generic(q, v, dvdx, dvdy, dvdxdy, xs, ys, xq, yq, xs_out, ys_out, extrapval);
}

// Far case (far because the derivatives are stored in separate arrays).
__global__ void getInterpolation2D_far(float2 * __restrict__ q,
                                             const float2 * __restrict__ v,
                                             const float2 * __restrict__ dvdx,
                                             const float2 * __restrict__ dvdy,
                                             const float2 * __restrict__ dvdxdy,
                                             const int xs, const int ys, const float * __restrict__ xq, const float * __restrict__ yq, const int xs_out, const int ys_out, const float2 extrapval)  {
    getInterpolation2D_far_generic(q, v, dvdx, dvdy, dvdxdy, xs, ys, xq, yq, xs_out, ys_out, extrapval);
}

// Real case.
__global__ void getInterpolation2D_far_real(float * __restrict__ q,
                                             const float * __restrict__ v,
                                             const float * __restrict__ dvdx,
                                             const float * __restrict__ dvdy,
                                             const float * __restrict__ dvdxdy,
                                             const int xs, const int ys, const float * __restrict__ xq, const float * __restrict__ yq, const int xs_out, const int ys_out, const float extrapval)  {
    getInterpolation2D_far_generic(q, v, dvdx, dvdy, dvdxdy, xs, ys, xq, yq, xs_out, ys_out, extrapval);
}