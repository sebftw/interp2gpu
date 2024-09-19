/*int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
if (i >= xs || j >= ys) {
    // Invalid threads.
    return;
}*/
int offset = 0;//ys * xs * (blockIdx.z * blockDim.z + threadIdx.z);  // Probably not working - have to add a full page (we need 4 coeffs as well, etc)
int idx = i + xs*j + offset;
int2 bounds = make_int2(xs, ys);
float2 query = make_float2(xq[idx], yq[idx])-1.0f; // Matlab uses 1-indexing
// float2 query = make_float2(__ldcg(xq+idx), __ldcg(yq+idx))-1.0f; // Matlab uses 1-indexing