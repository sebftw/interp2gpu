function Vq = interp2gpu(V, Xq, Yq, method, extrapval)
%INTERP2GPU interpolation for 2-D gridded gpuArray data, based on interp2.
%
%  [Vq] = interp2gpu(V, Xq, Yq, method) is equivalent to 
%  [Vq] = interp2(V, Xq, Yq, method, 0)
% 
% This means
% 1) No extrapolation is used. Outside values are set to zero.
% 2) Assumes a unit-spaced grid: X=1:N and Y=1:M where [M,N]=SIZE(V, 1:2).
%
% It supports an additional method on GPU:
% [Vq] = interp2gpu(V, Xq, Yq, 'spline_approx') fast approximate spline.
%
% Batching is supported for spline interpolation! A GPU is high-throughput,
% so interpolating a [M,N,K]=SIZE(V, 1:3) video will be much faster than
% processing each frame individually.

% TODO: We do not support all the call signatures of interp2 (e.g. Xq, Yq
% vectors). We could also add Makima.
% Version 0.5, Sebastian Kazmarek Præsius, 16 Sept., 2022.
% Version 1.0, Sebastian Kazmarek Præsius, 25 Oct., 2024.
%  Added support for real input.

narginchk(1, 5); % allowing for an ExtrapVal

if nargin < 4
    method = 'linear';
end

if nargin < 5
    extrapval = 0;
end

% Ensure gpuArray.
V = gpuArray(V);
Xq = gpuArray(Xq);
Yq = gpuArray(Yq);

if strcmpi(method, 'nearest') || ...
   strcmpi(method, 'linear')  || ...
   strcmpi(method, 'cubic')
    % Cases where we can just fall back to the built-in interp2.
    Vq = interp2(V, Xq, Yq, method, extrapval);
    return;
end

spline_approx = strcmpi(method, 'spline_approx');

if strcmp(method, 'spline') || spline_approx
    derivatives = gpuThomas2D(V, spline_approx, spline_approx);

    % The fastest format for evaluation is to pack the values and
    % derivatives into one matrix. For real-valued data, this would mean
    % packing {V, dVdx, dVdy, dVdxdy} in one matrix of float4.

    if numel(derivatives) == 3
       % The far format: derivatives = {dVdx, dVdy, dVdxdy};
       % TODO: Implement kernel that takes this format efficiently!
       %       For now just split it, and use the far-split kernel.
       if isreal(V)
           interpolation2D = get_kernel('getInterpolation2D_far_real');
       else
           interpolation2D = get_kernel('getInterpolation2D_far');
       end
    else
        % The far-split format: derivatives = {real(dVdx), imag(dVdx), real(dVdy), imag(dVdy), real(dVdxdy), imag(dVdxdy)}
        interpolation2D = get_kernel('getInterpolation2D_far_split');
    end

    % We allow batching over all the dimensions following the two first
    batch_size = prod(size(V, 3:ndims(V)));
    interpolation2D.GridSize(3) = batch_size;
    Vq = feval(interpolation2D, zeros(size(Xq), "like", V), V, derivatives{:}, size(V, 1), size(V, 2), Yq, Xq, size(Xq, 1), size(Yq, 2), extrapval);
else
    error('Interpolation method not supported on GPU');
end

end

