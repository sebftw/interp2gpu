function derivatives = gpuThomas2D(V, approx_dim1, approx_dim2)
    %GPUTHOMAS2D Computes the derivatives for 2D spline interpolation.
    %
    % [derivatives] = GPUTHOMAS2D(V)
    % Uses a linear system S, and computes S*V to get d/dy,
    % then V*S' to get d/dx, and finally S*V*S' to get d^2/dxdy.
    %
    % [derivatives] = GPUTHOMAS2D(V, approx_dim1)
    % If approx_dim1 == true, the values d/dy are approximated by a
    % convolution: conv2(V, s, 'same'), which can give a performance boost.
    %
    % Note: The system S is memoized to increase performance. This means
    % that GPU memory will leak if varying sized images are used.
    % Version 0.5, Sebastian Kazmarek Præsius, 16 Sept., 2022.

    % The default is not to approximate Y or X derivatives.
    if nargin<=1
        approx_dim1 = false;
    end
    if nargin<=2
        approx_dim2 = false;
    end

    % Set up cached objects. Instead of solving the matrix equations every
    % function call, we store the inverted matrix. GEMM is fast(er) on GPU.
    persistent getpreandc_memoized
    if isempty(getpreandc_memoized)
        getpreandc_memoized = memoize(@getpreandc);
    end

    % Retrieve memoized objects.
    if ~approx_dim1
        % No need to retrieve matrix if we are not going to use it.
        matr1 = getpreandc_memoized(size(V, 1));
    end
    if ~approx_dim2
        % No need to retrieve matrix if we are not going to use it.
        matr2 = getpreandc_memoized(size(V, 2));
    end
    
    % Similar for the approximation filter. Calculate once and retrieve
    persistent approx_filter
    if isempty(approx_filter)
        % An approximate filter, which computes the matrix product for
        % derivatives in the axial direction as a simple convolution.
        kk = 16+1; % 16 (single precision) or 24 (double precision)
        se = @(kk) (sqrt(3)/6)*(-2+sqrt(3)).^(abs((1:kk*2-1)-kk));
        te = conv(se(kk), [3, 0, -3],'valid');
        approx_filter = gpuArray(single(te));
    end

    % So many ways to right-multiply matrices in MATLAB:
    % 1) x * matr2'; (R2006a)
    % And when we want multiple outputs:
    % 2) pagefun(@mtimes, x, matr2'); (R2013b)
    % 3) pagemtimes(x, 'none', matr2, 'transpose'); (R2020b)
    % 4) gpucoder.batchedMatrixMultiply(R, matr2, dR1, matr2, 'transpose', 'NT');
    % 5) gpucoder.stridedMatrixMultiply(cat(4, R, dR1), matr2, 'transpose', 'NT'); (R2020a)
    % 6) tensorprod (R2022a)
    % (1) Is the fastest, but due to various overhead it may be slower when
    % executed in a loop. So use another method when batching!

    % For left-multiply we have an additional way of doing it by
    % matricizing the tensor, and just carrying out the product:
    % 7) reshape(matr1 * reshape(x, size(x, 1), []), size(x));
    % Then GEMM can be used instead of batched-GEMM, and still have batching!
    % Has same batched performance as equivalent option 2 and 3 (see below)

    % Comparison of their speed (higher=better):
    %            (1)     (2)     (3)
    % Unbatched: 1920    1820    1655
    %     Batch: 2000    2380    2340
    % We see (1) is best when not batched. It almost matches my earlier
    % mex-implementation, which had a rate of 1980 unbatched.
    % Options 3, 4, 5 were no faster than 2 when batched, but require much 
    % newer MATLAB versions. So use (1) unbatched and (2) for batched!
    
    % W/O memoization overhead, which is surprisingly large in MATLAB
    % compared to Python:
    %          (1)     (2)
    % Single: 2000    1900
    %  Batch: 2030    2380
    % It is apparent the overhead of just retrieving memoized matrices is
    % large! But when amortized over a batch it becomes essentially nothing.
    % It seems my mex-implementation is still faster by around 20%, unless
    % the filter approximation is applied, and they have about same speed.

    % When not batched or anything, right multiplication, (1), is simple.
    rmult = @(x) x * matr2';  % This is faster than pre-transposing. I guess MATLAB optimizes the matrix-transposematrix-product to a single BLAS call.
    if ~ismatrix(V)
        % The batched case. Use method (2) instead.
        rmult = @(x) pagefun(@mtimes, x, matr2');
    end
    if approx_dim2
        % Use approximation by convolution instead :)
        rmult = @(x) convn(x, approx_filter, 'same');
    end
    
    % Left-multiplication can always be done in a batched manner (7), not a
    % lot of people know this; it is generally faster in my experience!
    lmult = @(x) reshape(matr1 * reshape(x, size(x, 1), []), size(x));
    % ^ equal to @(x) matr1 * x when not batched.
    if approx_dim1
        % Approximation by convolution.
        lmult = @(x) convn(x, approx_filter', 'same');
    end

    if isreal(V) || (approx_dim1 && approx_dim2)
        % Simple case!
        [dVdx, dVdy, dVdxdy] = deriv(V, lmult, rmult);
        derivatives = {dVdx, dVdy, dVdxdy};
        return;
    end

    % Advanced cases, because:
    % It turns out to be fastest to split into real and complex parts,
    % and then just multiply each part separately by the real matrices
    % matr1 and matr2. The split is just 3.5% of time spent.
    % Probably because BLAS does not support real-times-complex
    % matrix products, and two real-times-real products are much faster.

    if approx_dim1
        % The case with axial derivatives approximation.
        dV1 = lmult(V);  % <- Complex.
        [R, I] = arrayfun(@split_complex, V);
        dR2 = rmult(R);
        dI2 = rmult(I);
        [dR1, dI1] = arrayfun(@split_complex, dV1);
        dR3 = rmult(dR1);
        dI3 = rmult(dI1);
    elseif approx_dim2
        error('approx dim2 without approx dim1 is not implemented yet!');
    else
        % The case with no approximation and complex input.
        % Split to real and imaginary parts.
        [R, I] = arrayfun(@split_complex, V);
        [dR1, dR2, dR3] = deriv(R, lmult, rmult);
        % clear R  % Save memory
        [dI1, dI2, dI3] = deriv(I, lmult, rmult);
    end
    % Now we have derivatives in FAR SPLIT format:
    derivatives = {dR1, dI1, dR2, dI2, dR3, dI3};

    % FAR because we store d/dx, d/dy, d^2/dxdy in separate tensors.
    % SPLIT because we separate the real and imaginary parts as well.
    % A terrible format in terms of cache locality. Results in abysmal
    % spline evaluation performance. But if it is only evaluated once,
    % it is not worth it to spend any time re-ordering the data in memory.
end

function [dfdx, dfdy, dfdxdy] = deriv(f, lmult, rmult)
    dfdx = lmult(f);
    dfdy = rmult(f);
    dfdxdy = rmult(dfdx);
end

function [r, i] = split_complex(C)
    r = real(C);
    i = imag(C);
end

function [prod, pre, C] = getpreandc(n)
    % GETPREANDC computes the system S used to get derivatives.
    % [prod, pre, C] = GETPREANDC(n) computes inv(C)*P for a system of size
    % n, and also returns P and C.
    % and C as additional output arguments.
    % Inspired by built-ins spline and parallel.internal.flowthrough.spline
    % (MATLAB 2021b). These find derivatives y' of y by solving: C*y'=pre*y
    if n < 4
        % Spline interpolation with not-a-knot end-conditions require four
        % points. Other cubic splines usually require at least three.
        error('Need at least four points in each dimension for spline interpolation.');
    end
    % Compute the matrix in double precision, and then turn it to single.
    pre = eye(n, 'double');
    prediff = diff(pre, [], 1);
    pre(1, :)=((1+4)*prediff(1, :)+prediff(2, :))/2;
    pre(2:n-1, :) = convn(prediff(1:n-1, :), [3; 3], 'valid');
    pre(n, :) = (prediff(n-2, :)+(1+4)*prediff(n-1, :))/2;
    
    C = spdiags([[ones(n-2, 1);2;0]   ...
                 [1;4*ones(n-2, 1);1] ...
                 [0;2;ones(n-2, 1)] ], [-1 0 1], n, n);
    prod = cast(full(double(C))\pre, 'like', zeros(1, 1, 'single', 'gpuArray'));
end