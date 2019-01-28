% settings
l1 = [25, 401];
l2 = [10, l1(1)+1];

% options
options = optimset('GradObj', 'on', 'MaxIter', 100);
lambda = 0.1;

% load training and testing datasets
[Xtrnt, ytrn] = readmnist('train-images-idx3-ubyte', 'train-labels-idx1-ubyte');
[Xtstt, ytst] = readmnist('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte');
ytrn(ytrn == 0) = 10;
ytst(ytst == 0) = 10;

% resize to 20x20 (including scaling to [0..1])
n = neuroelf;
fprintf('Resizing %d training images (. = 1000) ', numel(ytrn));
Xtrn = zeros(numel(ytrn), 400);
for c = 1:numel(ytrn)
    Xtrn(c, :) = reshape(n.image_resize((1 / 255) .* double(reshape( ...
        Xtrnt(c, :, :), [28, 28])), 20, 20), 1, 400);
    if mod(c, 1000) == 0
        fprintf('.');
    end
end
clear Xtrnt;
fprintf('\n');
pause(0.01);
fprintf('Resizing %d testing images (. = 1000) ', numel(ytst));
Xtst = zeros(numel(ytst), 400);
for c = 1:numel(ytst)
    Xtst(c, :) = reshape(n.image_resize((1 / 255) .* double(reshape( ...
        Xtstt(c, :, :), [28, 28])), 20, 20), 1, 400);
    if mod(c, 1000) == 0
        fprintf('.');
    end
end
clear Xtstt;
fprintf('\n');
pause(0.01);

% transpose X's
Xtrn = Xtrn';
Xtst = Xtst';

% generate Y's
Ytrn = zeros(max(ytrn), numel(ytrn));
Ytst = zeros(max(ytst), numel(ytst));
Ytrn(ytrn + (0:size(Ytrn, 1):size(Ytrn, 1)*(numel(ytrn)-1))') = 1;
Ytst(ytst + (0:size(Ytst, 1):size(Ytst, 1)*(numel(ytst)-1))') = 1;
if sum(Ytrn(:)) ~= numel(ytrn) || sum(sum(Ytrn==1, 1)) ~= numel(ytrn)
    error('Invalid assignment');
end
if sum(Ytst(:)) ~= numel(ytst) || sum(sum(Ytst==1, 1)) ~= numel(ytst)
    error('Invalid assignment');
end

% generate random t (with fixed values), [25, 401]; [10, 26]
s = RandStream('mt19937ar', 'Seed', 0);
RandStream.setGlobalStream(s);
t = 0.5 - rand(prod(l1) + prod(l2), 1);

% time training
costfun = @(theta)allcost(theta, Xtrn, Ytrn, lambda, {l1, l2});
tic
[final_t, Js, ex] = fmincg(costfun, t, options);
toc

% results
[J, Jg, p] = allcost(final_t, Xtst, Ytst, lambda, {l1,l2});
disp(1 - sum(p == ytst) / numel(ytst));
n = histcounts2(ytst, p);
disp(n);
