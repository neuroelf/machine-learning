% load X and y
load ex3data1.mat
load ex3weights.mat

% layer sizes
l1 = [25, size(X,2)+1];
l2 = [10, l1(1)+1];

% options
options = optimset('GradObj', 'on', 'MaxIter', 250);
lambda = 6.2e-4;

% create Y (i.e. Y')
Y = zeros(max(y), numel(y));
Y(y + (0:size(Y,1):((numel(y)-1)*size(Y,1)))') = 1;

% transpose X
X = X';

% generate random t (with fixed values), [25, 401]; [10, 26]
s = RandStream('mt19937ar', 'Seed', 0);
RandStream.setGlobalStream(s);
t = 0.5 - rand(prod(l1) + prod(l2), 1);

% time training
costfun = @(theta)allcost(theta, X, Y, lambda, {l1, l2});
tic
[final_t, Js, ex] = fmincg(costfun, t, options);
toc

% test given Theta values
tic
[given_t, gJs, gex] = fmincg(costfun, [Theta1(:); Theta2(:)], options);
toc
