function Vq = interp2gpu(V, Xq, Yq, method)
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
% vectors), and we should also verify that non-complex input works as well.
% Version 0.5, Sebastian Kazmarek Præsius, 16 Sept., 2022.

if nargin < 4
    method = 'linear';
end

% Ensure gpuArray.
V = gpuArray(V);
Xq = gpuArray(Xq);
Yq = gpuArray(Yq);

if strcmp(method, 'nearest') || ...
   strcmp(method, 'linear')  || ...
   strcmp(method, 'cubic')
    % Cases where we can just fall back to built-in interp2.
    Vq = interp2(V, Xq, Yq, method, 0);
    return;
end


spline_approx = strcmp(method, 'spline_approx');

if strcmp(method, 'spline') || spline_approx
    derivatives = gpuThomas2D(V, spline_approx, spline_approx);

    if ~iscell(derivatives)
        % In case the derivatives are all stored in a single matrix.
        % We had this when we used cuBLAS.
        interpolation2D = get_kernel('getInterpolation2D');
        Vq = feval(interpolation2D, V, derivatives, Yq, Xq, size(V, 1), size(V, 2));
    elseif numel(derivatives) == 3
        % The far format: derivatives = {dVdx, dVdy, dVdxdy};
        % TODO: Implement kernel that takes this format efficiently!
        %       For now just split it, and use the far-split kernel.
        derivatives = {real(derivatives{1}), imag(derivatives{1}), real(derivatives{2}), imag(derivatives{2}), real(derivatives{3}), imag(derivatives{3})};
        interpolation2D = get_kernel('getInterpolation2D_far_split');
        % We allow batching over all the dimensions following the two first
        sizeV = size(V);
        batch_size = prod(sizeV(3:ndims(V)));
        interpolation2D.GridSize(3) = batch_size;
        Vq = feval(interpolation2D, V, V, derivatives{1}, derivatives{2}, derivatives{3}, derivatives{4}, derivatives{5}, derivatives{6}, Yq, Xq, size(V, 1), size(V, 2));
    else
        % The far-split format: derivatives = {dR1, dI1, dR2, dI2, dR3, dI3}
        % This is the path we currently use.
        interpolation2D = get_kernel('getInterpolation2D_far_split');

        % We allow batching over all the dimensions following the two first
        sizeV = size(V);
        batch_size = prod(sizeV(3:ndims(V)));
        interpolation2D.GridSize(3) = batch_size;
        Vq = feval(interpolation2D, V, V, derivatives{1}, derivatives{2}, derivatives{3}, derivatives{4}, derivatives{5}, derivatives{6}, Yq, Xq, size(V, 1), size(V, 2));
    end
else
    error('Interpolation method not supported on GPU');
end

end

