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

/*
class split_complex {
public:
    const float* __restrict__ real;
    const float* __restrict__ imag;
    
    __host__ __device__ inline
    split_complex(const float* __restrict__ real, const float* __restrict__ imag) : real(real), imag(imag) {}
    // Retrieve the i'th element, as if this was a normal array.
    __host__ __device__ inline
    float2 operator [] (int i) const {return {real[i], imag[i]};}
};*/

inline __device__ float2 slerp(float2 a, float2 b, float t) {
    // Assumes the inputs are normalized vectors.
    // https://en.wikipedia.org/wiki/Slerp
    float angle = dot(a, b);
    angle = clamp(angle, -1.0f, 1.0f);  // Due to possible inaccuracies.
    angle = acosf(angle);
    
    if (angle < SMALL_ANGLE) {
        // in this case we have sin(angle) = angle, so no reason to do the heavy math below.
        return lerp(a, b, t); //make_float2(lerp(a.x, b.x, t), lerp(a.y, b.y, t));
        // There seems to be a problem in that if the movement is zero, we still get non-identical result
    }

    float c1 = sinf((1.0f-t)*angle);
    float c2 = sinf(t*angle);
    //angle = 1.0f/sinf(angle);
    //c1 = c1 * angle;
    //c2 = c2 * angle;
    //return make_float2(a.x*c1 + c2*b.x, a.y*c1 + c2*b.y);
    return normalize(a * c1 + b * c2);
}

template<typename interpT>
__device__ interpT do_interp2d(const interpT *v, const int2 p, const float2 xy, const int2 bounds, const int offset) {
    interpT f00   = vsub(1, p.x, p.y),
            fx00  = vsub(2, p.x, p.y),
            fy00  = vsub(3, p.x, p.y),
            fxy00 = vsub(4, p.x, p.y),
        
            f01   = vsub(1, p.x, p.y+1),
            fx01  = vsub(2, p.x, p.y+1),
            fy01  = vsub(3, p.x, p.y+1),
            fxy01 = vsub(4, p.x, p.y+1),
        
            f10   = vsub(1, p.x+1, p.y),
            fx10  = vsub(2, p.x+1, p.y),
            fy10  = vsub(3, p.x+1, p.y),
            fxy10 = vsub(4, p.x+1, p.y),
        
            f11   = vsub(1, p.x+1, p.y+1),
            fx11  = vsub(2, p.x+1, p.y+1),
            fy11  = vsub(3, p.x+1, p.y+1),
            fxy11 = vsub(4, p.x+1, p.y+1);
    
    float x = xy.x;
    float y = xy.y;
    float y2 = y * y;
    float x2 = x * x;  // another form without squaring may be nice, if at all possible.
    return ((-3 + 3 * y) * y2 + (2 - 2 * y) * y2 * x) * x2 * fy11 + ((3 + (-6 + 3 * y) * y) * y + (-2 + (4 - 2 * y) * y) * y * x) * x2 * fy10 + ((-1 + y) * y2 + ((3 - 3 * y) * y2 + (-2 + 2 * y) * y2 * x) * x2) * fy01 + ((1 + (-2 + y) * y) * y + ((-3 + (6 - 3 * y) * y) * y + (2 + (-4 + 2 * y) * y) * y * x) * x2) * fy00 + ((1 - y) * y2 + (-1 + y) * y2 * x) * x2 * fxy11 + ((-1 + (2 - y) * y) * y + (1 + (-2 + y) * y) * y * x) * x2 * fxy10 + ((-1 + y) * y2 + ((2 - 2 * y) * y2 + (-1 + y) * y2 * x) * x) * x * fxy01 + ((1 + (-2 + y) * y) * y + ((-2 + (4 - 2 * y) * y) * y + (1 + (-2 + y) * y) * y * x) * x) * x * fxy00 + ((-3 + 2 * y) * y2 + (3 - 2 * y) * y2 * x) * x2 * fx11 + (-1 + (3 - 2 * y) * y2 + (1 + (-3 + 2 * y) * y2) * x) * x2 * fx10 + ((3 - 2 * y) * y2 + ((-6 + 4 * y) * y2 + (3 - 2 * y) * y2 * x) * x) * x * fx01 + (1 + (-3 + 2 * y) * y2 + (-2 + (6 - 4 * y) * y2 + (1 + (-3 + 2 * y) * y2) * x) * x) * x * fx00 + ((9 - 6 * y) * y2 + (-6 + 4 * y) * y2 * x) * x2 * f11 + (3 + (-9 + 6 * y) * y2 + (-2 + (6 - 4 * y) * y2) * x) * x2 * f10 + ((3 - 2 * y) * y2 + ((-9 + 6 * y) * y2 + (6 - 4 * y) * y2 * x) * x2) * f01 + (1 + (-3 + 2 * y) * y2 + (-3 + (9 - 6 * y) * y2 + (2 + (-6 + 4 * y) * y2) * x) * x2) * f00;
}


__device__ float2 do_interp2d_split(const float *v_real, const float *v_imag, const int2 p, const float2 xy, const int2 bounds, const int offset) {
    float2  f00   = vsub_split(1, p.x, p.y),
            fx00  = vsub_split(2, p.x, p.y),
            fy00  = vsub_split(3, p.x, p.y),
            fxy00 = vsub_split(4, p.x, p.y),
        
            f01   = vsub_split(1, p.x, p.y+1),
            fx01  = vsub_split(2, p.x, p.y+1),
            fy01  = vsub_split(3, p.x, p.y+1),
            fxy01 = vsub_split(4, p.x, p.y+1),
        
            f10   = vsub_split(1, p.x+1, p.y),
            fx10  = vsub_split(2, p.x+1, p.y),
            fy10  = vsub_split(3, p.x+1, p.y),
            fxy10 = vsub_split(4, p.x+1, p.y),
        
            f11   = vsub_split(1, p.x+1, p.y+1),
            fx11  = vsub_split(2, p.x+1, p.y+1),
            fy11  = vsub_split(3, p.x+1, p.y+1),
            fxy11 = vsub_split(4, p.x+1, p.y+1);
    
    float x = xy.x;
    float y = xy.y;
    float y2 = y * y;
    float x2 = x * x;  // another form without squaring may be nice, if at all possible.
    return ((-3 + 3 * y) * y2 + (2 - 2 * y) * y2 * x) * x2 * fy11 + ((3 + (-6 + 3 * y) * y) * y + (-2 + (4 - 2 * y) * y) * y * x) * x2 * fy10 + ((-1 + y) * y2 + ((3 - 3 * y) * y2 + (-2 + 2 * y) * y2 * x) * x2) * fy01 + ((1 + (-2 + y) * y) * y + ((-3 + (6 - 3 * y) * y) * y + (2 + (-4 + 2 * y) * y) * y * x) * x2) * fy00 + ((1 - y) * y2 + (-1 + y) * y2 * x) * x2 * fxy11 + ((-1 + (2 - y) * y) * y + (1 + (-2 + y) * y) * y * x) * x2 * fxy10 + ((-1 + y) * y2 + ((2 - 2 * y) * y2 + (-1 + y) * y2 * x) * x) * x * fxy01 + ((1 + (-2 + y) * y) * y + ((-2 + (4 - 2 * y) * y) * y + (1 + (-2 + y) * y) * y * x) * x) * x * fxy00 + ((-3 + 2 * y) * y2 + (3 - 2 * y) * y2 * x) * x2 * fx11 + (-1 + (3 - 2 * y) * y2 + (1 + (-3 + 2 * y) * y2) * x) * x2 * fx10 + ((3 - 2 * y) * y2 + ((-6 + 4 * y) * y2 + (3 - 2 * y) * y2 * x) * x) * x * fx01 + (1 + (-3 + 2 * y) * y2 + (-2 + (6 - 4 * y) * y2 + (1 + (-3 + 2 * y) * y2) * x) * x) * x * fx00 + ((9 - 6 * y) * y2 + (-6 + 4 * y) * y2 * x) * x2 * f11 + (3 + (-9 + 6 * y) * y2 + (-2 + (6 - 4 * y) * y2) * x) * x2 * f10 + ((3 - 2 * y) * y2 + ((-9 + 6 * y) * y2 + (6 - 4 * y) * y2 * x) * x2) * f01 + (1 + (-3 + 2 * y) * y2 + (-3 + (9 - 6 * y) * y2 + (2 + (-6 + 4 * y) * y2) * x) * x2) * f00;
}

// #define vsub_split(C, X, Y)   make_float2(v_real[(X)+bounds.x*(Y)+bounds.x*bounds.y*(C-1)+offset], v_imag[(X)+bounds.x*(Y)+bounds.x*bounds.y*(C-1)+offset])
template <typename Tv, typename Tdv>
__device__ float2 do_interp2d_far_split(Tv v, Tdv dvdx, Tdv dvdy, Tdv dvdxdy, const int2 p, const float2 xy, const int2 bounds) {
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
    
    auto x = xy.x;
    auto y = xy.y;
    auto y2 = y * y;
    auto x2 = x * x;  // another form without squaring may be nice, if at all possible.
    return ((-3 + 3 * y) * y2 + (2 - 2 * y) * y2 * x) * x2 * fy11 + ((3 + (-6 + 3 * y) * y) * y + (-2 + (4 - 2 * y) * y) * y * x) * x2 * fy10 + ((-1 + y) * y2 + ((3 - 3 * y) * y2 + (-2 + 2 * y) * y2 * x) * x2) * fy01 + ((1 + (-2 + y) * y) * y + ((-3 + (6 - 3 * y) * y) * y + (2 + (-4 + 2 * y) * y) * y * x) * x2) * fy00 + ((1 - y) * y2 + (-1 + y) * y2 * x) * x2 * fxy11 + ((-1 + (2 - y) * y) * y + (1 + (-2 + y) * y) * y * x) * x2 * fxy10 + ((-1 + y) * y2 + ((2 - 2 * y) * y2 + (-1 + y) * y2 * x) * x) * x * fxy01 + ((1 + (-2 + y) * y) * y + ((-2 + (4 - 2 * y) * y) * y + (1 + (-2 + y) * y) * y * x) * x) * x * fxy00 + ((-3 + 2 * y) * y2 + (3 - 2 * y) * y2 * x) * x2 * fx11 + (-1 + (3 - 2 * y) * y2 + (1 + (-3 + 2 * y) * y2) * x) * x2 * fx10 + ((3 - 2 * y) * y2 + ((-6 + 4 * y) * y2 + (3 - 2 * y) * y2 * x) * x) * x * fx01 + (1 + (-3 + 2 * y) * y2 + (-2 + (6 - 4 * y) * y2 + (1 + (-3 + 2 * y) * y2) * x) * x) * x * fx00 + ((9 - 6 * y) * y2 + (-6 + 4 * y) * y2 * x) * x2 * f11 + (3 + (-9 + 6 * y) * y2 + (-2 + (6 - 4 * y) * y2) * x) * x2 * f10 + ((3 - 2 * y) * y2 + ((-9 + 6 * y) * y2 + (6 - 4 * y) * y2 * x) * x2) * f01 + (1 + (-3 + 2 * y) * y2 + (-3 + (9 - 6 * y) * y2 + (2 + (-6 + 4 * y) * y2) * x) * x2) * f00;
}

extern __shared__ float2 inside[];

__device__ float2 do_slerp2d(const float2 *f, const int2 p, const float2 xy, const int2 bounds, const int offset) {
    float2  omega00 = normalize(fsub(p.x, p.y)),
            omega01 = normalize(fsub(p.x, p.y+1)),
            omega10 = normalize(fsub(p.x+1, p.y)),
            omega11 = normalize(fsub(p.x+1, p.y+1));
    
    // TODO: Verify slerp formula, as an adaptation of bilinear interpolation.
    return slerp(slerp(omega00, omega10, xy.x), slerp(omega01, omega11, xy.x), xy.y);
    //return normalize(lerp(normalize(lerp(omega00, omega10, xy.x)), normalize(lerp(omega01, omega11, xy.x)), xy.y));
}

template<typename interpT>
__device__ interpT getP(const interpT *v, float2 &query, int2 &bounds, const int offset) {
    // Matlab uses 1-indexing
    int2 p = make_int2(floorf(query.x), floorf(query.y));

    p.x = p.x - (p.x>=bounds.x-1 ? 1 : 0);  // In case we are exactly on the image edge, take one step back (use x=1 instead of x=0 in this edge case).
    p.y = p.y - (p.y>=bounds.y-1 ? 1 : 0);
    if ((query.x < 0) || (query.y < 0) || (query.x > bounds.x) || (query.y > bounds.y)) {
        // Extrapolation outside of image: Set to zero.
        return interpT();
    }
    
    // Move the query to relative inside interval: [0, 1]
    query.x -= p.x;
    query.y -= p.y;
    
    interpT q = do_interp2d(v, p, query, bounds, offset);

    // Store the offset position of the interval in the bounds variable.
    bounds.x = p.x;
    bounds.y = p.y;
    return q;
}

__device__ float2 getP_split(const float *v_real, const float *v_imag, float2 &query, int2 &bounds, const int offset) {
    // Matlab uses 1-indexing
    int2 p = make_int2(floorf(query.x), floorf(query.y));
    
    p.x = p.x - (p.x>=bounds.x-1 ? 1 : 0);  // In case we are exactly on the image edge, take one step back (use x=1 instead of x=0 in this edge case).
    p.y = p.y - (p.y>=bounds.y-1 ? 1 : 0);
    if ((query.x < 0) || (query.y < 0) || (query.x > bounds.x-1) || (query.y > bounds.y-1)) {
        // Extrapolation outside of image: Set to zero.
        return make_float2(0.0f, 0.0f);
    }
    
    // Move the query to relative inside interval: [0, 1]
    query.x -= p.x;
    query.y -= p.y;
    
    float2 q = do_interp2d_split(v_real, v_imag, p, query, bounds, offset);

    // Store the offset position of the interval in the bounds variable.
    bounds.x = p.x;
    bounds.y = p.y;
    return q;
}

template <typename Tv, typename Tdv>
__device__ float2 getP_far_split(Tv v, Tdv dvdx, Tdv dvdy, Tdv dvdxdy, float2 &query, int2 &bounds, const int offset) {
    // Matlab uses 1-indexing
    int2 p = make_int2(floorf(query.x), floorf(query.y));
    
    p.x = p.x - (p.x>=bounds.x-1 ? 1 : 0);  // In case we are exactly on the image edge, take one step back (use x=1 instead of x=0 in this edge case).
    p.y = p.y - (p.y>=bounds.y-1 ? 1 : 0);
    if ((query.x < 0) || (query.y < 0) || (query.x > bounds.x-1) || (query.y > bounds.y-1)) {
        // Extrapolation outside of image: Set to zero.
        return make_float2(0.0f, 0.0f);
    }
    
    // Move the query to relative inside interval: [0, 1]
    query.x -= p.x;
    query.y -= p.y;
    
    float2 q = do_interp2d_far_split(v, dvdx, dvdy, dvdxdy, p, query, bounds);

    // Store the offset position of the interval in the bounds variable.
    bounds.x = p.x;
    bounds.y = p.y;
    return q;
}

__global__ void getInterpolation2D_special(float2 * q, const float2 * f, const float * v, const float * xq, const float * yq, const int xs, const int ys)  {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ys; j += blockDim.y * gridDim.y)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < xs; i += blockDim.x * gridDim.x) {
            #include "getInterpolation2D_snippet.cu"
            int2 p = make_int2(bounds.x, bounds.y);  // To retrieve p.
            float qv = getP(v, query, p, offset);
            if (qv == 0.0f)
                // If image absolute value is zero, not reason to find phase.
                q[idx] = make_float2(0.0f, 0.0f);
            else
                // TODO: Verify slerp formula, as an adaptation of bilinear interpolation.
                q[idx] = qv * do_slerp2d(f, p, query, bounds, offset);
        }
}

__global__ void getInterpolation2D(float2 * q, const float2 * v, const float * xq, const float * yq, const int xs, const int ys)  {
    
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ys; j += blockDim.y * gridDim.y)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < xs; i += blockDim.x * gridDim.x) {
            #include "getInterpolation2D_snippet.cu"
            q[idx] = getP(v, query, bounds, offset);
        }
}

__global__ void getInterpolation2D_split(float2 * q, const float * v_real, const float * v_imag, const float * xq, const float * yq, const int xs, const int ys)  {
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ys; j += blockDim.y * gridDim.y)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < xs; i += blockDim.x * gridDim.x) {
            #include "getInterpolation2D_snippet.cu"
            q[idx] = getP_split(v_real, v_imag, query, bounds, offset);
        }
}


__global__ void getInterpolation2D_far_split(float2 * __restrict__ q,
                                             const float2 * __restrict__ v,
                                             const float * __restrict__ dvdx_real, const float * __restrict__ dvdx_imag,
                                             const float * __restrict__ dvdy_real, const float * __restrict__ dvdy_imag,
                                             const float * __restrict__ dvdxdy_real, const float * __restrict__ dvdxdy_imag,
                                             const float * __restrict__ xq, const float * __restrict__ yq, const int xs, const int ys)  {
    // This is the function that we currently use for 2D interpolation.

    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    // First offset to get the page this block must work on.
    int batch_idx = blockIdx.z;
    int page_size = xs * ys;
    int page = batch_idx * page_size;
    q += page;
    v += page;
    
    dvdx_real += page;
    dvdx_imag += page;
    dvdy_real += page;
    dvdy_imag += page;
    dvdxdy_real += page;
    dvdxdy_imag += page;
    yq += page;
    xq += page;
    // This shifting to the specific page costs quite a few registers no?
    // We could store the pointers in shared memory if it is detrimental.
    
    auto dvdx = split_complex<const float * __restrict__, float2>(dvdx_real, dvdx_imag);
    auto dvdy = split_complex<const float * __restrict__, float2>(dvdy_real, dvdy_imag);
    auto dvdxdy = split_complex<const float * __restrict__, float2>(dvdxdy_real, dvdxdy_imag);

    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ys; j += blockDim.y * gridDim.y)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < xs; i += blockDim.x * gridDim.x) {
            #include "getInterpolation2D_snippet.cu"
            q[idx] = getP_far_split(v, dvdx, dvdy, dvdxdy, query, bounds, offset);
        }
}

__global__ void getInterpolation2D_far_split2(float2 * __restrict__ q,
                                             const float2 * __restrict__ v,
                                             const float * __restrict__ dvdx_real, const float * __restrict__ dvdx_imag,
                                             const float * __restrict__ dvdy_real, const float * __restrict__ dvdy_imag,
                                             const float * __restrict__ dvdxdy_real, const float * __restrict__ dvdxdy_imag,
                                             const float * __restrict__ xq, const float * yq, const int xs, const int ys)  {
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    // First offset to get the page this block must work on.
    int batch_idx = blockIdx.z/2;
    q += xs * ys * batch_idx;
    v += xs * ys * batch_idx;
    
    dvdx_real += xs * ys * batch_idx;
    dvdx_imag += xs * ys * batch_idx;
    dvdy_real += xs * ys * batch_idx;
    dvdy_imag += xs * ys * batch_idx;
    dvdxdy_real += xs * ys * batch_idx;
    dvdxdy_imag += xs * ys * batch_idx;  // Costs quite a few registers?
    yq += xs * ys * batch_idx;
    xq += xs * ys * batch_idx;
    
    auto dvdx = split_complex<const float * __restrict__, float2>(dvdx_real, dvdx_imag);
    auto dvdy = split_complex<const float * __restrict__, float2>(dvdy_real, dvdy_imag);
    auto dvdxdy = split_complex<const float * __restrict__, float2>(dvdxdy_real, dvdxdy_imag);

    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ys; j += blockDim.y * gridDim.y)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < xs; i += blockDim.x * gridDim.x) {
            #include "getInterpolation2D_snippet.cu"
            // Write to the two pages.
            auto query_save=query;
            auto bounds_save=bounds;
            q[idx] = getP_far_split(v, dvdx, dvdy, dvdxdy, query, bounds, offset);
            q[idx+xs*ys] = getP_far_split(v+xs*ys, dvdx+xs*ys, dvdy+xs*ys, dvdxdy+xs*ys, query_save, bounds_save, offset);
        }
}
