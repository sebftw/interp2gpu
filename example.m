%% An arbitrary image to test the implementation with.
img = double(rgb2gray(imread('ngc6543a.jpg')));

rng(101);
img = img + rand(size(img)) * .5; % Add a little noise to remove quantization effects.
img = min(max(img, 0), 255) / 255;  % Convert to values in [0, 1] for display.

%% First we want to check that our implementation is correct.

% Diplay original image.
f = figure(1);
f.Name = "Comparison of Implementations";
subplot(131);
imshow(img); axis image;
title("Original Image"); 

% Now we want to sample it differently.
y = 1:size(img, 1); y = y';
x = 1:size(img, 2);

[X, Y] = meshgrid(x, y); % Create all pixel positions.

% Let's try with a -60 degree rotation around the center and 1.5x zoom.
theta = -60 * pi/180;
zoom = 1.5;
mu = [mean(x([1, end])); mean(y([1, end]))];
[X, Y] = arrayfun(getrotationzoom(theta, mu, zoom), X, Y);
% Lets do a little more manipulations.
X = rot90(X);
Y = rot90(Y);
Y(end-100:end, :) = Y(end-100:end, :) * 0 + y(end-100:end);
X(end-100:end, 1:end-numel(x)-1) = 0;
X(end-100:end, end-(numel(x)-1:-1:2)) = X(end-100:end, 2:numel(x)-1) * 0 + x(2:end-1);

% Use as refence a double-precision interp2 (MATLAB) implementation.
reference = interp2(double(img), X, Y, "spline", 0);

% Now compute the other two results using the exact same inputs.
X = single(X); Y = single(Y); img = single(img);
groundtruth = interp2(img, X, Y, "spline", 0);
proposed = interp2gpu(img, X, Y, "spline", 0);
proposed_approx = interp2gpu(img, X, Y, "spline_approx", 0);

ax1 = subplot(132);
imshow(groundtruth); axis image;
title("Resampled (Ground Truth)");

ax2 = subplot(133);
imshow(proposed); axis image;
title("Resampled (GPU Implementation)");
linkaxes([ax1, ax2]);

%% Now to check correctness more quantitatively.
% The functions interp2 and interp2gpu took the same input, so their errors
% should be distributed the same since it is just rounding errors.

flatten = @(x) x(:); 
maxdeviation = @(x) max(abs(x), [], "all");
todb = @(x) 20 * log10(abs(x));

reference_db0 = todb(maxdeviation(reference));

error_groundtruth = groundtruth - reference; % Get ground truth error distribution.
error_proposed = proposed - reference;

% Now plot the distribution of relative errors in dB.
nbins = 201;
BinLimits = [-250, 0];
BinLimits(2) = todb(max([max(abs(error_groundtruth), [], "all"), max(abs(error_proposed), [], "all")])) - reference_db0;  % No reason to make bins where there is no data.
[error_groundtruth_dist, edges] = histcounts(todb(error_groundtruth) - reference_db0, nbins, "BinLimits", BinLimits, "Normalization", "pdf");
[error_proposed_dist] = histcounts(todb(error_proposed) - reference_db0, edges, "Normalization", "pdf");

f = figure(2);
f.Name = "Investigation of Errors";
plot(edges(1:end-1), error_proposed_dist, "-", "DisplayName", "Proposed GPU implementation");
hold on;
plot(edges(1:end-1), error_groundtruth_dist, "-", "DisplayName", " MATLAB's interp2");
hold off;
title("Distribution of numerical errors.");
xlabel("Error magnitude [dB]");
ylabel("Probability density");
grid on;
legend;
xlim([BinLimits(1), 0]);
drawnow;
% They have the same distribution of errors, therefore the implementation
% is correct (any error in implementation would increase the overall error,
% leading to a difference in distribution between the two.)

%% Lastly, inter2gpu supports a "spline approximation by convolution".
% Just to show that this approximation is quite accurate in practice.

yidx = ceil(size(Y, 1) * 0.5);  % Show error in the middle row.

f = figure(3);
f.Name = "Fast Approximated Spline";
ax1 = subplot(221);
imshow(proposed); axis image;
title({"Resampled (GPU Implementation)", "Exact"});
yline(yidx, "Color","red");

ax2 = subplot(222);
imshow(proposed_approx); axis image;
title({"Resampled (GPU Implementation)", "Approximated"});
yline(yidx, "Color","red");
linkaxes([ax1, ax2]);


subplot(223);
[error_proposed_approx_dist] = histcounts(todb(proposed_approx - reference) - reference_db0, edges, "Normalization", "pdf");
figure(3);
% plot(edges(1:end-1), error_groundtruth_dist, "DisplayName", "MATLAB's interp2");
p1 = plot(edges(1:end-1), error_proposed_dist, "-", "DisplayName", "interp2gpu");
hold on;
p2 = plot(edges(1:end-1), error_proposed_approx_dist, "--", "DisplayName", "interp2gpu (Approximated)");
hold off;
title("Distribution of numerical errors.");
xlabel("Error magnitude [dB]");
ylabel("Probability density");
legend("Location", "SouthOutside");
xlim([BinLimits(1), 0]);

ax = subplot(224);
relative = @(x) x(yidx, :);
% relative = @(x) abs(relative(x));  % Uncomment to show absolute error.
% relative = @(x) relative(x)./abs(reference(yidx, :));   % Uncomment to show relative error.
plot(relative(proposed - reference), "-",  "DisplayName", "interp2gpu");
hold on;
plot(relative(proposed_approx - reference), "--",  "DisplayName", "interp2gpu (approximated)");
hold off;
ax.XAxis.Color = "red";
ax.YAxis.Color = "red";
ax.YAxisLocation = "right";

if size(Y, 1) > 100
    xlim([1, 100]);
end
xlabel("Pixel # (horizontally)");
ylabel("Error");
% legend;
title("Error in the middle row of the image.", "Color", "red");


%% Measure the processing time.

disp(['Measuring performance for ' gpuDevice().Name '.']);

disp(" Single-image interpolation.");
imggpu = gpuArray(img); Xgpu = gpuArray(X); Ygpu = gpuArray(Y);
t_interp2 = timeit(@() interp2(img, X, Y, "spline", 0), 1);
t_interp2gpu = gputimeit(@() interp2gpu(imggpu, Xgpu, Ygpu, "spline", 0), 1);
disp("  Rate for interp2:    " + num2str(1/t_interp2, 4) + "  images/s");
disp("  Rate for interp2gpu: " + num2str(1/t_interp2gpu, 4) + " images/s");
disp("  Speed-up factor: " + num2str(t_interp2/t_interp2gpu, 2));

% To fully occupy all cores (multiprocessors) in the GPU, the image either
% needs to be large, or there needs to be many images. interp2gpu therefore
% supports giving it many images at once in a batch.

disp(' ');
disp(" Multiple-image interpolation.");
batch_size = 20;
Xb = repmat(X, 1, 1, batch_size);
Yb = repmat(Y, 1, 1, batch_size);
imgb = repmat(img, 1, 1, batch_size);
imgbgpu = gpuArray(imgb); Xbgpu = gpuArray(Xb); Ybgpu = gpuArray(Yb);

t_interp2_batch = timeit(@() interp2loop(Xb, Yb, imgb), 1);
t_interp2gpu_batch = gputimeit(@() interp2gpu(imgbgpu, Xbgpu, Ybgpu, "spline", 0), 1);
disp("  Rate for interp2:   " + num2str(batch_size/t_interp2_batch, 4) + "  images/s");
disp("  Rate for interp2gpu: " + num2str(batch_size/t_interp2gpu_batch, 4) + " images/s");
disp("  Speed-up factor: " + num2str(t_interp2_batch/t_interp2gpu_batch, 4));

disp(' ');
disp(" Complex-valued interpolation (for multiple images).");
imgb_c = imgb + 1i * rand(size(imgb), "like", imgb);
imgbgpu_c = gpuArray(imgb_c);

t_interp2_batch_complex = timeit(@() interp2loop(Xb, Yb, imgb_c), 1);
t_interp2gpu_batch_complex = gputimeit(@() interp2gpu(imgbgpu_c, Xbgpu, Ybgpu, "spline", 0), 1);
disp("  Rate for interp2:    " + num2str(batch_size/t_interp2_batch_complex, 4) + "  images/s");
disp("  Rate for interp2gpu: " + num2str(batch_size/t_interp2gpu_batch_complex, 4) + " images/s");
disp("  Speed-up factor: " + num2str(t_interp2_batch_complex/t_interp2gpu_batch_complex, 4));
disp(' ');
disp(" Approximated interpolation (for multiple complex-valued images).");

t_interp2gpu_batch_complex_approx = gputimeit(@() interp2gpu(imgbgpu_c, Xbgpu, Ybgpu, "spline_approx", 0), 1);
disp("  Rate for interp2:    " + num2str(batch_size/t_interp2_batch_complex, 4) + "  images/s");
disp("  Rate for interp2gpu: " + num2str(batch_size/t_interp2gpu_batch_complex_approx, 4) + " images/s");
disp("  Speed-up factor: " + num2str(t_interp2_batch_complex/t_interp2gpu_batch_complex_approx, 4));

disp(" Approximated interpolation (for multiple real-valued images).");
t_interp2gpu_batch_real_approx = gputimeit(@() interp2gpu(imgbgpu, Xbgpu, Ybgpu, "spline_approx", 0), 1);
disp("  Rate for interp2:    " + num2str(batch_size/t_interp2_batch, 4) + "  images/s");
disp("  Rate for interp2gpu: " + num2str(batch_size/t_interp2gpu_batch_real_approx, 4) + " images/s");
disp("  Speed-up factor: " + num2str(t_interp2_batch/t_interp2gpu_batch_real_approx, 4));

disp('Finished measuring performance.');

%% Plot speed-up chart.
f = figure(4);
f.Name = "Comparison of Performance";
bardata = [[t_interp2_batch, t_interp2_batch]./[t_interp2gpu_batch, t_interp2gpu_batch_real_approx]; [t_interp2_batch_complex, t_interp2_batch_complex]./[t_interp2gpu_batch_complex, t_interp2gpu_batch_complex_approx]];
b = bar(bardata');
% xticks(b.XEndPoints(:));
xticks([1, 2]);
xticklabels([{"Exact spline"}, {"Approximated spline"}]);
legend({"Real input", "Complex input"}, "Location", "NorthWest");
grid on;
% title("Measured performance-boost from using GPU (Real Inputs).");
ylabel("How many times faster interp2gpu was.");
title("GPU implementation was >" + num2str(round(floor(min(bardata(:, 1))))) + "x faster");

function result = interp2loop(Xb, Yb, imgb)
    result = zeros(size(Xb), "like", imgb);
    for i = 1:size(imgb, 3)
        result(:, :, i) = interp2(imgb(:, :, i), Xb(:, :, i), Yb(:, :, i), "spline", 0);
    end
end


function fun = getrotationzoom(theta, mu, zoom)
    R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    R = R / zoom;
    fun = @applyrotation;
    function [x, y] = applyrotation(x, y)
        xy = [x; y];
        xy = R * (xy - mu) + mu;
        x = xy(1);
        y = xy(2);
    end
end