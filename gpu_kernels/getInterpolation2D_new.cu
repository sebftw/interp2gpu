#include <cuda_runtime.h>
#include "helper_math.h"

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

template<typename interpT>
__device__ float2 getP(const interpT * __restrict__ v, float2 &query, int2 &bounds) {

    int2 p = make_int2(floor(query.x), floor(query.y));
    
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
    return q;
}


// const Ty * __restrict__ xq, const Ty * __restrict__ yq
template <typename Td, typename Ty2>
__global__ void getInterpolation2D(Td * __restrict__ q, const Td * __restrict__ v, const Ty2 query, const int xs, const int ys)  {
    int offset = threadIdx.z;  // <- Allow for batched interpolation.
    // Uses 2D grid stride pattern, for spatial locality in both x and y and to not have to do integer division.
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    
    // So xq and yq are in a far format.
    // 
    
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < ys; j += blockDim.y * gridDim.y)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < xs; i += blockDim.x * gridDim.x) {
            int idx = (offset * ys + j) * xs + i; // i + xs*j + ys*offset;
			int2 bounds = make_int2(xs, ys);
			//auto qu = {query[idx].x-1, query[idx].y-1}; // Matlab uses 1-based indexing
			
			// Make sure predication is done, or at least that there is not branching which prevents unrolling, as that would be a shame.
			if(!(0 <= query.x & query.x <= xs-1 & 0 <= query.y & query.y <= ys-1)) {
				// Values outside of the defined area: Do not interpolate, just set to zero.
				q[idx] = Td();
				continue;
			}
			
            // q[idx] = getP(v, query, bounds, offset);
        }
}



__global__ void getInterpolation2D(float2 * __restrict__ out, const float2 * __restrict__ v, const float * __restrict__ xq, const float * __restrict__ yq, const int xs, const int ys) {
	getInterpolation2D(out, v, split_complex<const float * __restrict__, float2>(xq, yq), xs, ys);
}
