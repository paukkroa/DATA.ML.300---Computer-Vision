% Copyright (C) Andrea Vedaldi and Andrew Zisserman
%
% The purpose of this exercise is to observe different building-blocks of
% Convolutional Neural Networks (CNN), and use Stochastic Gradient Descent 
% (SGD) to train a CNN.
%
% Fill any required parts with your own code, and answer any questions
% asked in each section.


%% Run setup before continuing
% Use ctrl+Enter (or click 'Run Section') to run each section separately
setup ;


%% Example image
% Read an example image
x = imread('peppers.png') ;

% Convert to single format for MatConvNet
x = im2single(x) ;

% Visualize the input x
figure(1) ; clf ; imagesc(x) ;

% Your task: Use MATLAB's 'size' function to display the size of x.
%            What is the size of third dimension and why?

%%% Your code starts here %%%
size(x)
%%% Your code stops here %%%

% The size is 384, 512, 3. The size of the third dimension is therefore 3
% which relates to the three color channels of an image (R,G,B).
% There are individual intensity values for each colors, which allow
% creating any visible color by adjusting the ratios of the different
% intensities

%% Creating a filter bank
% Create a bank of 10 linear filters with size 5x5x3
w = randn(5,5,3,10,'single') ;  % again, single precision

% Apply the convolutional operator
y = vl_nnconv(x, w, []) ;  % the 3rd argument here is a vector of bias terms (empty in our case)

% Visualize the output y
figure(2) ; clf ; vl_imarraysc(y) ; colormap gray ;

% Try running this section a few times. 

% Your task: Is there any difference between each run? Why?
%            What is the size of the output y, and how is this related to
%            x and w?

% The results differ slightly since the filters in the filter bank are randomized each time
% this cell is ran.
size(y)
size(w)
% The size of y is 380, 503, 10. The first two dimensions are related to
% the image size. In each edge 2 pixels are cutoff since they
% are combined using the filter kernel w into the pixel on the "3rd"
% row. As this happens with all edges of the original image, it will reduce
% the dimensions by 2*2, which is 4 pixels in x and y dimensions
% respectively.



%% Applying downsampling and padding
% Try again, downsampling the output
y_ds = vl_nnconv(x, w, [], 'stride', 16);
figure(3); clf; vl_imarraysc(y_ds); colormap gray; title('Downsampling');

% Try (zero)padding
y_pad = vl_nnconv(x, w, [], 'pad', 2);
figure(4); clf; vl_imarraysc(y_pad); colormap gray; title('Padding');

% Your task: How does the size of y_pad differ from previous y? Can you explain why?
size(y)
size(y_pad)
% The y_pad is now back at the same size in the first two
% dimensions as the original image. There is a padding of size 2 added in
% the convolution network, which adds 2 pixels of padding each edge, which
% then comes back to the original dimension of the image.

%% Manually design a filter
 
%%% Your code starts here %%%
w2 = [0  -1  0;
      -1  4  -1;
      0  -1  0];
%%% Your code stops here %%%

w2 = repmat(w2, [1, 1, 3]);

w2 = single(w2) ;  % single conversion
y_lap = vl_nnconv(x, w2, []) ;
figure(5) ; clf ; colormap gray ;

subplot(1,3,1) ; imagesc(x) ; title('Input');
subplot(1,3,2) ; imagesc(y_lap) ; title('filter output') ;
subplot(1,3,3) ; imagesc(-abs(y_lap)) ; title('- abs(filter output)');

% Your task: Currently the filter does nothing to the input image. 
%            Replace w2 with a 3x3 implementation of the Laplacian
%            operator. 
%            Why is the repmat function needed here?
%            Take a look at the result. 
%            What kind of a features does our filter extract?
      
% The repmat function converts the 1D matrix w2 into a 3D matrix, so it can
% be used for all channels (R,G & B).
% Our filter acts as an edge detection algorithm. When the image is
% inversed the edges are clearly visible

%% Non-linear gating (ReLU)

% Create a filter
w = single(repmat([1 0 -1], [1, 1, 3]));  
w = cat(4, w, -w); 

% Apply convolution
y = vl_nnconv(x, w, []);

% Non-linear activation function
z = vl_nnrelu(y);  % vl_nnrelu function implements ReLU

figure(6); clf; colormap gray;
subplot(1,2,1); vl_imarraysc(y); title('Filter output');
subplot(1,2,2); vl_imarraysc(z); title('ReLU output');

% Your task: Some of the functions in a CNN should be non-linear. Why?

% Some of the functions need to be non-linear to introduce non-linearity
% into the model. With only linear functions it is not possible to estimate
% linear models, thus we need non-linearity to reproduce more complex
% models. Non-linearity allows us to find more complex features in the
% data, such as distincting different objects and faces for example.
% In this case, the ReLU allows us to find the edges really clearly, where
% as the filter doesn't quite have a similar contrast to the background.

%% Pooling
y = vl_nnpool(x, 15, 'Stride', 4) ;  % max pooling with a square filter of size 15
figure(7) ; clf ;
subplot(1,2,1); imagesc(x); title('No pooling');
subplot(1,2,2); imagesc(y); title('Max-pooling');

% Your task: Compare the result of max-pooling to the original.
%            What is the effect of max-pooling? 
%            What does the 'Stride' parameter do?

% Max-pooling clearly distorts the image. It seems to have also increased
% the brightness of the image, with not that many dark areas compared to
% the original. The image quality has also decreased. As the size of the
% kernel is clearly larger than the stride size, the output image has also
% fused together values from multiple sampling position, resulting in
% slighlty changed shapes as well (take a look at the onion in the front
% right for example)
% 
% The stride parameter adjusts the "step-size" of the kernel movements. In
% this case, it is set to 4, which means that the kernel moves 4 pixels
% each time.

%% Implementing a small CNN and optimizing with SGD
%  We will train a CNN to extract blob-like structures from an image.

% 1. Start by running the algorithm without any pre-processing.
%    The blue lines in the histograms of scores represent classification 
%    thresholds, where values are either classifed as positive hits 
%    (those belonging to blobs) or negative hits. Values between these two 
%    thresholds are ignored.
%    How would the histograms set in an ideal case?
%    What is the result here compared to the ideal case?

% In the ideal case the histograms would set into a stepped function like shape, 
% for example with the positive hits it could approximate a function like y = {1 if
% x>1, else 0}. This would mean a perfect classification accuracy with
% positive hits. Same for negative hits, but with function like y = {1 if
% x<0, else 0}. If looking at the histogram from a far, it
% would look like a U, with negative hits below 0 and positive hits above
% 1.
% The result here resembles a rising function with a heavy fluctuation.
% Also, there seems to be a moderate amount of false negatives

% 2. Train the tiny CNN by first smoothing the input image and subtracting 
%    the median value in preprocessing. Use the imsmooth function 
%    (defined in imsmooth.m) with the sigma value of 3 for smoothing.
%    The learned filter should resemble the discretisation of a well-known differential operator. 
%    Which one? 

% The filter clearly resembles a Laplacian operator, since there is a large
% concentration of negative numbers near the center, and more positive
% numbers on the outside. This could make it easier for the CNN to detect
% objects by the edges. Also, the histogram looks way better now.

% 3. Try doubling the learning rate.
%    What is the effect of having too high of a learning rate?
%    Restore the learning rate and set momentum to 0.
%    How does this differ from the previous with the same learning rate?
%    What is the benefit of using momentum?

% A higher learning rate tends to oscillate the error/accuracy rates, as
% the gradients might be too large and they "hop" over the local minimums.
% This makes it harder for the network to converge and it can be seen in
% the amount of misclassifications and histograms.
%
% Setting the momentum to 0 reduces the speed at which the network is
% converging. The decrease in error rate is almost stagnant near the final
% iterations. Momentum definitely allows for a faster convergence while
% keeping the gradients manageable.


% Load an image
im = rgb2gray(im2single(imread('data/dots.jpg'))) ;

% Compute the location of black blobs in the image
[pos,neg] = extractBlackBlobs(im) ;

fig = figure('Name','test', 'Position', [0,0,1000,600]);


% Pre-processing
%%% Your code starts here %%%
im = imsmooth(im, 3); % Smooth the image
im_med = median(im); % Calculate median value
im = im - im_med; % Subtract median value from image
%%% Your code ends here %%%


% Learning with stochastic gradient descent (SGD)

% SGD parameters:
% - numIterations: maximum number of iterations
% - rate: learning rate
% - momentum: momentum rate
% - shrinkRate: shrinkage rate5(or coefficient of the L2 regulariser)
% - plotPeriod: how often to plot

numIterations = 500 ;
rate = 5 ;
momentum = 0 ;
shrinkRate = 0.0001 ;
plotPeriod = 10 ;

% Initial CNN parameters:
w = 10 * randn(5, 5, 1) ;
w = single(w - mean(w(:))) ;
b = single(0) ;

% Create pixel-level labes to compute the loss
y = zeros(size(pos),'single') ;
y(pos) = +1 ;
y(neg) = -1 ;

% Initial momentum
w_momentum = zeros('like', w) ;
b_momentum = zeros('like', b) ;

% SGD with momentum
for t = 1:numIterations

  % Forward pass
  res = tinycnn(im, w, b) ;

  % Loss
  z = y .* (res.x3 - 1) ;

  E(1,t) = ...
    mean(max(0, 1 - res.x3(pos))) + ...
    mean(max(0, res.x3(neg))) ;
  E(2,t) = 0.5 * shrinkRate * sum(w(:).^2) ;
  E(3,t) = E(1,t) + E(2,t) ;

  dzdx3 = ...
    - single(res.x3 < 1 & pos) / sum(pos(:)) + ...
    + single(res.x3 > 0 & neg) / sum(neg(:)) ;

  % Backward pass
  res = tinycnn(im, w, b, dzdx3) ;

  % Update momentum
  w_momentum = momentum * w_momentum + rate * (res.dzdw + shrinkRate * w) ;
  b_momentum = momentum * b_momentum + rate * 0.1 * res.dzdb ;

  % Gradient step
  w = w - w_momentum ;
  b = b - b_momentum ;
  
  % Plots
  if mod(t-1, plotPeriod) == 0 || t == numIterations
    fp = res.x3 > 0 & y < 0 ;
    fn = res.x3 < 1 & y > 0 ;
    tn = res.x3 <= 0 & y < 0 ;
    tp = res.x3 >= 1 & y > 0 ;
    err = cat(3, fp|fn , tp|tn, y==0) ;
   
    set(0, 'currentfigure', fig); clf;
    colormap gray ;

    subplot(2,3,1) ;
    plot(1:t, E(:,1:t)') ;
    grid on ; title('objective') ;
    ylim([0 1.5]) ; legend('error', 'regularizer', 'total') ;

    subplot(2,3,2) ; hold on ;
    [h,x]=hist(res.x3(pos(:)),30) ; plot(x,h/max(h),'g') ;
    [h,x]=hist(res.x3(neg(:)),30) ; plot(x,h/max(h),'r') ;
    plot([0 0], [0 1], 'b--') ;
    plot([1 1], [0 1], 'b--') ;
    xlim([-2 3]) ;
    title('histograms of scores') ; legend('pos', 'neg') ;

    subplot(2,3,3) ;
    vl_imarraysc(w) ;
    title('learned filter') ; axis equal ;

    subplot(2,3,4) ;
    imagesc(res.x3) ;
    title('network output') ; axis equal ;

    subplot(2,3,5) ;
    imagesc(res.x2) ;
    title('first layer output') ; axis equal ;

    subplot(2,3,6) ;
    image(err) ;
    title('red: false, green: correct, blue: ignore') ;

    if verLessThan('matlab', '8.4.0')
      drawnow ;
    else
      drawnow expose ;
    end
  end
end