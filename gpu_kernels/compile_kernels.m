% Script to compile CUDA kernels to ptx and mex files.
% It attempts to use the NVCC compiler shipped with MATLAB to minimize the
% maintenance needed (especially for mex code), making it self-contained.
%
% Version 1.0, May 1 2022, Sebastian Kazmarek Præsius
% Version 1.2, Feb 22 2023, Sebastian Kazmarek Præsius
%   Changed handling of PATH variable, and non-MATLAB-shipped CUDA.
% Version 1.3, Oct 10 2024, Sebastian Kazmarek Præsius
%   Added support for Windows.

% We have tried to generally use only files shipped with MATLAB, such that
% this compilation will work even if the user has not installed CUDA, but
% has only installed the MATLAB Parallel Toolbox.

nvcc_flags = '-Xptxas=-warn-spills';

% Put in your cuda installation path (if search paths did not work).
cuda_path = '';

% Search for CUDA.
search_paths = {cuda_path, getenv('CUDA_PATH'), ...
    fullfile(matlabroot, 'sys', 'cuda', computer('arch'), 'cuda'), ...
    getenv('CUDA_HOME'), fullfile(getenv('MW_NVCC_PATH'), '..'), ...
    fullfile(getenv('CUDA_BIN_PATH'), '..'), '/usr/local/cuda'};

for path = search_paths
    path = path{1};
    nvcc_path = fullfile(path, 'bin', 'nvcc');
    if ispc
        nvcc_path = [nvcc_path '.exe'];
    end

    if ~isempty(path) && logical(exist(nvcc_path, 'file'))
        cuda_path = path;
        break;
    end
end

assert(logical(exist(nvcc_path, 'file')), 'Could not find CUDA installation.');

% Some fixes for MATLAB-shipped nvcc ptx compilation.
includes = {fullfile(matlabroot, 'extern', 'include'), fullfile(matlabroot, 'toolbox', 'parallel', 'gpu', 'extern', 'include', 'gpu'), fullfile(cuda_path, 'include')};
libraries = {};
nvcc_flags_ptx = '';
cicc_dir = fullfile(cuda_path, 'nvvm', 'bin');
path_set = '';
is_builtin_cuda = contains(cuda_path, matlabroot); % <- if MATLAB-shipped CUDA was used.
if is_builtin_cuda
    libdevice_dir = fullfile(matlabroot, 'bin', computer('arch'));
    
    libraries = [libraries, cicc_dir];
    
    nvcc_flags_ptx = [nvcc_flags_ptx ' --dont-use-profile -ldir "' libdevice_dir '"'];

    % Add cicc to PATH environment variable.
    env_path = getenv('PATH');
    if ~contains(env_path, cicc_dir)
        % Add cicc to the path of child processes (for PTX compilation).
        env_path = [cicc_dir pathsep env_path];
        if ispc
            setenv('PATH', env_path);
        end
    end

    if ispc
        path_set = ['set "PATH=' cicc_dir pathsep fullfile(cuda_path, 'bin') pathsep '%PATH%" && '];
    else
        path_set = ['export PATH="' cicc_dir pathsep fullfile(cuda_path, 'bin') pathsep '$PATH" ; '];
    end

    % Mexcuda probably also applies these "fixes", which it loads from XML files in:
    % fullfile(matlabroot,'toolbox','parallel','gpu','extern','src','mex',computer('arch'))
    % unfortunately mexcuda does not seem to support .ptx compilation, and the
    % XML parser used is not open source, so we had to roll out our own solution.
else
    % The paths, includes and libraries are automatically set by the
    % profile in fullfile(cuda_path, 'bin', 'nvcc.profile').    
end

if ispc
    % Windows can be tricky. The easiest is if they have GCC installed with
    % Cygwin, so we can pretend it is Linux. Here is a fix if they instead
    % use Visual Studio for their compilation, to ensure it finds cl.exe.
    
    vspath = [getenv('ProgramFiles(x86)') '\Microsoft Visual Studio\Installer\vswhere.exe'];
    found  = exist(vspath, "file");
    if not(found)
        vspath = [getenv('ProgramFiles(x86)') '\Microsoft Visual Studio\Installer\vswhere.exe'];
        found = exist(vspath, "file");
    end
    
    if not(found)
        warning("Could not find your compiler (You are using Windows, but maybe Microsoft Visual Studio is not installed?).");
    end
    
    [status, cmdout] = system(vspath);
    cmdout = split(cmdout, newline);  % Split each line.

    searchstr = 'installationPath: ';
    vspath = cmdout{arrayfun(@(s) startsWith(s, searchstr), cmdout)};
    vspath = [vspath(numel(searchstr)+1:end) '\VC\Auxiliary\Build\vcvarsall.bat'];
    found = exist(vspath, "file");
    if not(found)
        error("Could not configure compilation on Windows.");
    end
    
    % nvcc_flags = [nvcc_flags ' -allow-unsupported-compiler'];
    path_set = ['"' vspath '" ' getenv("PROCESSOR_ARCHITECTURE") ' &&'];
end


strfun = @(x, pre) cell2mat(join(cellfun(@(y) [pre '"' y '"'], x, 'UniformOutput', false), ' '));
additional_flags = [strfun(includes, '-I') ' ' strfun(libraries, '-L') ' '];
nvcc_flags_ptx = [nvcc_flags_ptx ' ' additional_flags];


% Setup compilation functions.
compile_nvcc = @(name, varargin) cell2mat(join({path_set, ['"', nvcc_path, '"'], nvcc_flags, nvcc_flags_ptx, varargin{:}, name}));
compile_ptx = @(name, varargin) system(compile_nvcc(['-ptx ' name], varargin{:}));
compile_cubin = @(name, varargin) system(compile_nvcc(['-cubin ' name], varargin{:}));
compile_mex = @(name, varargin) mexcuda('-R2018a', ['NVCC_FLAGS="' nvcc_flags ' -Wno-deprecated-gpu-targets"'], varargin{:}, name);
compile_graph = @(name, entry, varargin) system([compile_nvcc([' -cubin ' name '.cu'], varargin{:}) ' && nvdisasm -cfg -g -gp -poff -fun ' num2str(entry) ' ' name '.cubin | dot -o' name num2str(entry) '.png -Tpng']);
% Compile graph takes a the name of a .cu file, without prefix, then generates a graph for the entry given as the second argument.
% Mex is needed in case we want to use libraries or API such as cuBLAS.

% Compile the processing kernels.
disp('Compiling CUDA kernels.');
compile_ptx('getInterpolation2D.cu');

disp('Done compiling kernels.');
