function result = get_kernel(kernel_name, reload)
%GET_KERNEL is a cache of CUDA kernels. Memoizes kernels used in various
% modules, since initializing a parallel.gpu.CUDAKernel is expensive.
%
% GET_KERNEL(kernel_name) returns the kernel with the given name.
%
% GET_KERNEL(kernel_name, reload)
% If a change is made to the .ptx code, kernels must be reloaded to reflect this.
% If reload=true, kernels are reloaded. Short form is: GET_KERNEL('', 1).

% Version 1.0, May 1 2022, Sebastian Kazmarek Præsius

% The reason to not use the built-in "memoize" function was that memoize
% returns a local map object, but we want the map to be globally available,
% as well as handle the setup of the map.

if nargin == 1
    reload = false;
end

% To prevent loading kernels every time they are used, we cache them.
% When they are loaded the first time they are compiled for the GPU, much like a shader, and this is expensive.
persistent kernel_cache

if ~isempty(kernel_cache) && isKey(kernel_cache, kernel_name) && ~reload
    % ^ Put && false to force re-load of all kernels if a change is made.

    % Load kernel from cache.
    result = kernel_cache(kernel_name);
    return;
end

% Use paths relative to this script's location.
current_dir = pwd;
program_path = mfilename('fullpath');
program_path = [program_path(1:end-numel(mfilename)) 'gpu_kernels'];
cd(program_path);

% Now we can safely use relative paths from the dir of this script.
if ~exist('getInterpolation2D.ptx', 'file') || reload
    % We could switch to using Makefiles if more complexity ensues.
    % Currently, if you update a kernel, just re-compile it manually.
    compile_kernels;
end

% Load and configure kernel launch parameters.
D = gpuDevice;
kernel_cache = containers.Map();

% This GPU has full occupancy at 48 warps per SM.
% This means 48 * 32 * D.MultiprocessorCount = 125952 threads is needed to have full occupancy.
threads = 128; % 128 or 256
% With 128 threads per block, we then would like 125952/threads = 984 blocks.

kern = parallel.gpu.CUDAKernel('getInterpolation2D.ptx','getInterpolation2D.cu', 'getInterpolation2D_far_split');
kern.ThreadBlockSize = [32 16];
kern.GridSize = [12 D.MultiprocessorCount];
kernel_cache('getInterpolation2D_far_split') = kern;

kern = parallel.gpu.CUDAKernel('getInterpolation2D.ptx','getInterpolation2D.cu', 'getInterpolation2D_far');
kern.ThreadBlockSize = [32 16];
kern.GridSize = [12 D.MultiprocessorCount];
kernel_cache('getInterpolation2D_far') = kern;

kern = parallel.gpu.CUDAKernel('getInterpolation2D.ptx','getInterpolation2D.cu', 'getInterpolation2D_far_real');
kern.ThreadBlockSize = [32 16];
kern.GridSize = [12 D.MultiprocessorCount];
kernel_cache('getInterpolation2D_far_real') = kern;

cd(current_dir); % cd back to previous current dir.

if ~isempty(kernel_name)
    % Return kernel.
    % result = kernel_cache(kernel_name);
    result = get_kernel(kernel_name);
else
    disp(['Loaded kernels: ' strjoin(keys(kernel_cache), ', ')]);
end

end

