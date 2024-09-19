% Script to compile CUDA kernels to ptx and mex files.
% It attempts to use the NVCC compiler shipped with MATLAB to minimize the
% maintenance needed (especially for mex code), making it self-contained.
%
% Mex can contain C/C++, PTX, and possibly lower level GPU assembly. It is
% platform specific, and possibly also MATLAB-version specific.
%
% PTX is an intermediate assembly for a virtual GPU architecture, which is
% then compiled to the specific GPU by its driver the first time it is ran.
% Therefore it is way more portable.
%
% Version 1.0, May 1 2022, Sebastian Kazmarek Præsius
% Version 1.2, Feb 22 2023, Sebastian Kazmarek Præsius
%   Changed handling of PATH variable, and non-MATLAB-shipped CUDA.

% We have tried to generally use only files shipped with MATLAB, such that
% this compilation will work even if the user has not installed NVCC, but
% has only installed the MATLAB Parallel Toolbox.

nvcc_flags = '-lineinfo -Xptxas=-warn-spills -DN_TX_POS_MAX=512';
% -src-in-ptx

% Put in the correct cuda path (if search paths does not work).
cuda_path = '';
% cuda_path = '/usr/local/cuda-12.0/';
%cuda_path = getenv('CUDA_PATH');
% cuda_path = fullfile(matlabroot, 'sys', 'cuda', computer('arch'), 'cuda');

% Search for CUDA. First try user defined search path, then built-in MATLAB
% and afterwards, if still not found, try some obvious places to search.
search_paths = {cuda_path, fullfile(matlabroot, 'sys', 'cuda', computer('arch'), 'cuda'), ...
    getenv('CUDA_PATH'), getenv('CUDA_HOME'), fullfile(getenv('MW_NVCC_PATH'), '..'), ...
    fullfile(getenv('CUDA_BIN_PATH'), '..'), '/usr/local/cuda'};

for path = search_paths
    path = path{1};
    nvcc_path = fullfile(path, 'bin', 'nvcc');

    if ~isempty(path) && logical(exist(nvcc_path, 'file'))
        cuda_path = path;
        break;
    end
end

assert(logical(exist(nvcc_path, 'file')), 'Could not find CUDA installation.');
% ^ Will break on ealier MATLAB versions.
% We should provide fallback pre-compiled ptx for this case.

is_builtin_cuda = contains(cuda_path, matlabroot); % <- if MATLAB-shipped CUDA was used.

% Set the target architecture for PTX code.
if false
    D = gpuDevice;
    compute_capability = str2num(D.ComputeCapability);
    if compute_capability >= 8
        % Default is 52, and PTX is forward compatible but may be sub-optimal.
        % Specify the newest architecture possible for best performance.
        nvcc_flags = [nvcc_flags ' -arch=sm_86'];
        % nvcc_flags = [nvcc_flags ' -code=compute_52 -arch=compute_80'];
    end
end

% Some fixes for MATLAB-shipped nvcc ptx compilation.
includes = {fullfile(matlabroot, 'extern', 'include'), fullfile(matlabroot, 'toolbox', 'parallel', 'gpu', 'extern', 'include', 'gpu'), fullfile(cuda_path, 'include')};
libraries = {};
nvcc_flags_ptx = nvcc_flags;
cicc_dir = fullfile(cuda_path, 'nvvm', 'bin');
path_set = '';
if is_builtin_cuda
    libdevice_dir = fullfile(matlabroot, 'bin', computer('arch'));
    
    libraries = [libraries, cicc_dir];
    
    nvcc_flags_ptx = [nvcc_flags_ptx ' --dont-use-profile -ldir "' libdevice_dir '"'];

    % Add cicc to PATH environment variable.
    env_path = getenv('PATH');
    if ~contains(env_path, cicc_dir)
        % Add cicc to the path of child processes (PTX compilation).
        env_path = [cicc_dir pathsep env_path];
        % disp(env_path);
        if ispc
            % setenv('PATH', env_path);
        end
    end
    if ispc
        path_set = ['set "PATH=' cicc_dir pathsep fullfile(cuda_path, 'bin') pathsep '%PATH%" && '];
    else
        path_set = ['export PATH="' cicc_dir pathsep fullfile(cuda_path, 'bin') pathsep '$PATH" ; '];
    end
else
    % The paths, includes and libraries are automatically set by the
    % profile in fullfile(cuda_path, 'bin', 'nvcc.profile').
    % %ProgramFiles(x86)%\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64
    
end

strfun = @(x, pre) cell2mat(join(cellfun(@(y) [pre '"' y '"'], x, 'UniformOutput', false), ' '));
additional_flags = [strfun(includes, '-I') ' ' strfun(libraries, '-L') ' '];
nvcc_flags_ptx = [nvcc_flags_ptx ' ' additional_flags];

% Mexcuda probably also apply these "fixes", which it loads from XML files in:
% fullfile(matlabroot,'toolbox','parallel','gpu','extern','src','mex',computer('arch'))
% unfortunately mexcuda does not seem to support .ptx compilation, and the
% XML parser used is not open source, so we had to roll out our own code.


compile_nvcc = @(name, varargin) cell2mat(join({path_set, ['"', nvcc_path, '"'], nvcc_flags_ptx, varargin{:}, name}));

compile_ptx = @(name, varargin) system(compile_nvcc(['-ptx ' name], varargin{:}));
compile_cubin = @(name, varargin) system(compile_nvcc(['-cubin ' name], varargin{:}));
compile_mex = @(name, varargin) mexcuda('-R2018a', ['NVCC_FLAGS="' nvcc_flags ' -Wno-deprecated-gpu-targets"'], varargin{:}, name);
compile_graph = @(name, entry, varargin) system([compile_nvcc([' -cubin ' name '.cu'], varargin{:}) ' && nvdisasm -cfg -g -gp -poff -fun ' num2str(entry) ' ' name '.cubin | dot -o' name num2str(entry) '.png -Tpng']); % dot -ocfg.png -Tpng
% Compile graph takes a the name of a .cu file, without prefix, then
% generates a graph for the entry given as the second argument.
% compile_graph('dualstage_rcbf', '1', '-arch=sm_80 --use_fast_math -res-usage --nvlink-options=--verbose --nvlink-options=--dump-callgraph -DN_TX_POS_MAX=128');

% Mex is needed in case we want to use libraries such as cuBLAS, or API.

% Compile the processing kernels.
disp('Compiling CUDA kernels.');
compile_ptx('getInterpolation2D.cu');

disp('Done compiling kernels.');
